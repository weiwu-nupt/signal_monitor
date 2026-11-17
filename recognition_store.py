"""PostgreSQL recognition journal, with a SQLite compatibility backend for tests."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid
import zlib
from urllib.parse import quote, unquote, urlparse

import numpy as np

POSTGRES_CONFIG = Path(__file__).resolve().parent / 'results' / 'postgresql.json'
POSTGRES_DEFAULTS = dict(host='127.0.0.1', port=5432,
                         database='signal_monitor', user='postgres', password='')


def postgres_url(settings):
    """Build a correctly escaped DSN without exposing it in UI text."""
    user = quote(str(settings['user']), safe='')
    password = str(settings.get('password') or '')
    auth = user + (':'+quote(password, safe='') if password else '')
    return (f"postgresql://{auth}@{settings['host']}:{int(settings['port'])}/"
            f"{quote(str(settings['database']), safe='')}")


def database_settings():
    """Read local PostgreSQL settings; environment DSN has priority at runtime."""
    settings = dict(POSTGRES_DEFAULTS)
    try:
        if POSTGRES_CONFIG.exists():
            settings.update(json.loads(POSTGRES_CONFIG.read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError):
        pass
    return settings


def save_database_settings(settings):
    values = {key: settings[key] for key in POSTGRES_DEFAULTS}
    values['port'] = int(values['port'])
    if not values['host'] or not values['database'] or not values['user']:
        raise ValueError('数据库主机、名称和用户名不能为空')
    POSTGRES_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    temporary = POSTGRES_CONFIG.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, POSTGRES_CONFIG)


def database_url():
    # Explicit environment configuration remains useful for servers and CI.
    return os.environ.get('SIGNAL_MONITOR_DATABASE_URL') or postgres_url(database_settings())


DEFAULT_DATABASE = database_url()


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def model_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class _SQLiteRecognitionStore:
    """Own one connection on its worker thread; concurrent readers use WAL."""
    def __init__(self, path=DEFAULT_DATABASE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=5)
        try:
            self.connection.execute('PRAGMA foreign_keys=ON')
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, started_utc TEXT NOT NULL,
                    source TEXT NOT NULL, model_path TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL, settings_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS packets (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    created_utc TEXT NOT NULL, signal_id INTEGER,
                    frame INTEGER, cycle INTEGER, source_start_sample INTEGER,
                    batch INTEGER, channel_type TEXT, channel_slot INTEGER,
                    center_hz REAL NOT NULL, bandwidth_hz INTEGER NOT NULL,
                    sample_rate_hz INTEGER NOT NULL, sample_count INTEGER NOT NULL,
                    class_name TEXT NOT NULL, confidence REAL NOT NULL,
                    vote_fraction REAL NOT NULL, image_count INTEGER NOT NULL,
                    tied INTEGER NOT NULL, window_ms REAL NOT NULL, nfft INTEGER NOT NULL,
                    votes_json TEXT NOT NULL, probabilities_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    iq_path TEXT
                );
                CREATE TABLE IF NOT EXISTS windows (
                    packet_id INTEGER NOT NULL REFERENCES packets(id) ON DELETE CASCADE,
                    image_index INTEGER NOT NULL, start_sample INTEGER NOT NULL,
                    sample_count INTEGER NOT NULL, class_name TEXT NOT NULL,
                    confidence REAL NOT NULL, probabilities_json TEXT NOT NULL,
                    PRIMARY KEY (packet_id, image_index)
                );
                CREATE INDEX IF NOT EXISTS packets_signal ON packets(session_id,signal_id,id);
                CREATE INDEX IF NOT EXISTS packets_frequency ON packets(center_hz,created_utc);
            ''')
            columns = {row[1] for row in self.connection.execute('PRAGMA table_info(packets)')}
            if 'iq_path' not in columns:
                self.connection.execute('ALTER TABLE packets ADD COLUMN iq_path TEXT')
        except BaseException:
            self.connection.close()
            raise

    def start_session(self, source, model_path, settings, model_sha256=None):
        identifier = str(uuid.uuid4())
        digest = model_digest(model_path) if model_sha256 is None else model_sha256
        with self.connection:
            self.connection.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?)',
                (identifier, datetime.now(timezone.utc).isoformat(), str(source),
                 str(model_path), digest, _json(settings)))
        return identifier

    def _save_resampled_iq(self, packet_id, iq, sample_rate_hz):
        """Persist the exact DDC/resampled IQ behind a packet record."""
        if hasattr(iq, 'detach'):
            iq = iq.detach().to('cpu').numpy()
        iq = np.ascontiguousarray(np.asarray(iq))
        if iq.ndim != 2 or iq.shape[1] != 2 or iq.dtype != np.int16:
            raise ValueError('保存的重采样IQ必须为 Nx2 int16')
        relative = Path('iq_packets') / f'packet-{packet_id}.npz'
        target = self.path.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, iq=iq, sample_rate_hz=np.int64(sample_rate_hz))
        os.replace(temporary, target)
        with self.connection:
            self.connection.execute('UPDATE packets SET iq_path=? WHERE id=?',
                                    (str(relative), packet_id))
        return str(relative)

    def save(self, session_id, result, metadata, iq=None):
        if not result.windows:
            raise ValueError('packet has no per-image results')
        values = (session_id, datetime.now(timezone.utc).isoformat(),
                  metadata.get('signal_id'), metadata.get('frame'), metadata.get('cycle'),
                  metadata.get('source_start_sample'), metadata.get('batch'),
                  metadata.get('channel_type'), metadata.get('channel_slot'),
                  metadata['center_hz'], metadata['bandwidth_hz'],
                  result.sample_rate_hz, result.sample_count, result.class_name,
                  result.confidence, result.vote_fraction, len(result.windows),
                  int(result.tied), result.window_ms, result.nfft,
                  _json(result.vote_counts), _json(result.probabilities), _json(metadata))
        # Serialisation and both inserts belong to the same transaction. Never
        # leave a summary without its image votes if a write/encoding fails.
        with self.connection:
            cursor = self.connection.execute('''INSERT INTO packets
                (session_id,created_utc,signal_id,frame,cycle,source_start_sample,batch,
                 channel_type,channel_slot,center_hz,bandwidth_hz,sample_rate_hz,
                 sample_count,class_name,confidence,vote_fraction,image_count,tied,
                 window_ms,nfft,votes_json,probabilities_json,metadata_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', values)
            packet_id = cursor.lastrowid
            self.connection.executemany('INSERT INTO windows VALUES (?,?,?,?,?,?,?)',
                ((packet_id, i, w.start_sample, w.sample_count, w.class_name,
                  w.confidence, _json(w.probabilities)) for i, w in enumerate(result.windows, 1)))
        if iq is not None:
            self._save_resampled_iq(packet_id, iq, result.sample_rate_hz)
        return packet_id

    def recent(self, limit=200):
        self.connection.row_factory = sqlite3.Row
        return [dict(row) for row in self.connection.execute('''SELECT p.id,
            s.source,p.created_utc,p.signal_id,p.center_hz,p.bandwidth_hz,
            p.sample_rate_hz,p.class_name,p.confidence,p.vote_fraction,p.image_count,
            p.tied FROM packets p JOIN sessions s ON s.id=p.session_id
            ORDER BY p.id DESC LIMIT ?''', (limit,))]

    def query(self, sql, parameters=()):
        self.connection.row_factory = sqlite3.Row
        return [dict(row) for row in self.connection.execute(sql, parameters)]

    def load_iq(self, packet_id):
        row = self.connection.execute(
            'SELECT iq_path,sample_rate_hz FROM packets WHERE id=?', (packet_id,)).fetchone()
        if row is None or not row[0]:
            raise ValueError('该记录没有保存重采样 IQ')
        path = self.path.parent / row[0]
        with np.load(path, allow_pickle=False) as saved:
            return np.asarray(saved['iq'], dtype=np.int16), int(saved['sample_rate_hz'])

    def close(self):
        self.connection.close()


class _PostgreSQLRecognitionStore:
    """Network-safe record store.  IQ is retained in PostgreSQL BYTEA data."""
    def __init__(self, dsn):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError as exc:
            raise RuntimeError('缺少 PostgreSQL 驱动：请安装 psycopg2-binary') from exc
        self._psycopg2 = psycopg2
        self._cursor_factory = RealDictCursor
        self.dsn = str(dsn)
        self.connection = psycopg2.connect(self.dsn, connect_timeout=5)
        try:
            self._create_schema()
        except BaseException:
            self.connection.close()
            raise

    def _create_schema(self):
        with self.connection, self.connection.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id UUID PRIMARY KEY, started_utc TIMESTAMPTZ NOT NULL,
                    source TEXT NOT NULL, model_path TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL, settings_json JSONB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS packets (
                    id BIGSERIAL PRIMARY KEY,
                    session_id UUID NOT NULL REFERENCES sessions(id),
                    created_utc TIMESTAMPTZ NOT NULL,
                    signal_id INTEGER, frame INTEGER, cycle INTEGER,
                    source_start_sample BIGINT, batch INTEGER,
                    channel_type TEXT, channel_slot INTEGER,
                    center_hz DOUBLE PRECISION NOT NULL, bandwidth_hz BIGINT NOT NULL,
                    sample_rate_hz BIGINT NOT NULL, sample_count INTEGER NOT NULL,
                    class_name TEXT NOT NULL, confidence DOUBLE PRECISION NOT NULL,
                    vote_fraction DOUBLE PRECISION NOT NULL, image_count INTEGER NOT NULL,
                    tied BOOLEAN NOT NULL, window_ms DOUBLE PRECISION NOT NULL,
                    nfft INTEGER NOT NULL, votes_json JSONB NOT NULL,
                    probabilities_json JSONB NOT NULL, metadata_json JSONB NOT NULL,
                    iq_data BYTEA, iq_sample_count INTEGER
                );
                CREATE TABLE IF NOT EXISTS windows (
                    packet_id BIGINT NOT NULL REFERENCES packets(id) ON DELETE CASCADE,
                    image_index INTEGER NOT NULL, start_sample BIGINT NOT NULL,
                    sample_count INTEGER NOT NULL, class_name TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL, probabilities_json JSONB NOT NULL,
                    PRIMARY KEY (packet_id, image_index)
                );
                CREATE INDEX IF NOT EXISTS packets_signal ON packets(session_id, signal_id, id);
                CREATE INDEX IF NOT EXISTS packets_frequency ON packets(center_hz, created_utc);
            ''')

    def start_session(self, source, model_path, settings, model_sha256=None):
        # psycopg2 does not adapt uuid.UUID unless its optional adapter is
        # registered.  PostgreSQL safely casts this canonical UUID string to
        # the UUID column without global driver state.
        identifier = str(uuid.uuid4())
        digest = model_digest(model_path) if model_sha256 is None else model_sha256
        with self.connection, self.connection.cursor() as cursor:
            cursor.execute('''INSERT INTO sessions
                (id,started_utc,source,model_path,model_sha256,settings_json)
                VALUES (%s,%s,%s,%s,%s,%s::jsonb)''',
                (identifier, datetime.now(timezone.utc), str(source), str(model_path), digest, _json(settings)))
        return identifier

    @staticmethod
    def _iq_bytes(iq):
        if hasattr(iq, 'detach'):
            iq = iq.detach().to('cpu').numpy()
        iq = np.ascontiguousarray(np.asarray(iq))
        if iq.ndim != 2 or iq.shape[1] != 2 or iq.dtype != np.int16:
            raise ValueError('保存的重采样IQ必须为 Nx2 int16')
        return zlib.compress(iq.tobytes(), level=1), len(iq)

    def save(self, session_id, result, metadata, iq=None):
        if not result.windows:
            raise ValueError('packet has no per-image results')
        encoded_iq, iq_count = (None, None) if iq is None else self._iq_bytes(iq)
        values = (session_id, datetime.now(timezone.utc), metadata.get('signal_id'),
                  metadata.get('frame'), metadata.get('cycle'), metadata.get('source_start_sample'),
                  metadata.get('batch'), metadata.get('channel_type'), metadata.get('channel_slot'),
                  metadata['center_hz'], metadata['bandwidth_hz'], result.sample_rate_hz,
                  result.sample_count, result.class_name, result.confidence, result.vote_fraction,
                  len(result.windows), bool(result.tied), result.window_ms, result.nfft,
                  _json(result.vote_counts), _json(result.probabilities), _json(metadata),
                  self._psycopg2.Binary(encoded_iq) if encoded_iq is not None else None, iq_count)
        with self.connection, self.connection.cursor() as cursor:
            cursor.execute('''INSERT INTO packets
                (session_id,created_utc,signal_id,frame,cycle,source_start_sample,batch,
                 channel_type,channel_slot,center_hz,bandwidth_hz,sample_rate_hz,sample_count,
                 class_name,confidence,vote_fraction,image_count,tied,window_ms,nfft,
                 votes_json,probabilities_json,metadata_json,iq_data,iq_sample_count)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s::jsonb,%s::jsonb,%s::jsonb,%s,%s) RETURNING id''', values)
            packet_id = cursor.fetchone()[0]
            cursor.executemany('''INSERT INTO windows
                (packet_id,image_index,start_sample,sample_count,class_name,confidence,probabilities_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)''',
                [(packet_id, i, window.start_sample, window.sample_count,
                  window.class_name, window.confidence, _json(window.probabilities))
                 for i, window in enumerate(result.windows, 1)])
        return packet_id

    def query(self, sql, parameters=()):
        # Queries in the UI use SQLite's ? placeholder; make them portable.
        sql = sql.replace('?', '%s')
        with self.connection.cursor(cursor_factory=self._cursor_factory) as cursor:
            cursor.execute(sql, parameters)
            return [dict(row) for row in cursor.fetchall()]

    def recent(self, limit=200):
        return self.query('''SELECT p.id,s.source,p.created_utc,p.signal_id,p.center_hz,p.bandwidth_hz,
            p.sample_rate_hz,p.class_name,p.confidence,p.vote_fraction,p.image_count,p.tied
            FROM packets p JOIN sessions s ON s.id=p.session_id ORDER BY p.id DESC LIMIT %s''', (limit,))

    def load_iq(self, packet_id):
        rows = self.query('SELECT iq_data,iq_sample_count,sample_rate_hz FROM packets WHERE id=%s', (packet_id,))
        if not rows or rows[0]['iq_data'] is None:
            raise ValueError('该记录没有保存重采样 IQ')
        row = rows[0]
        raw = np.frombuffer(zlib.decompress(bytes(row['iq_data'])), dtype=np.int16)
        return raw.reshape(int(row['iq_sample_count']), 2), int(row['sample_rate_hz'])

    def close(self):
        self.connection.close()


class RecognitionStore:
    """Select PostgreSQL for normal operation; paths retain SQLite test support."""
    def __init__(self, database=None):
        self.database = database_url() if database is None else database
        self._impl = (_PostgreSQLRecognitionStore(self.database)
                      if isinstance(self.database, str) and self.database.startswith(('postgres://', 'postgresql://'))
                      else _SQLiteRecognitionStore(self.database))

    @property
    def connection(self):
        return self._impl.connection

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def start_session(self, *args, **kwargs): return self._impl.start_session(*args, **kwargs)
    def save(self, *args, **kwargs): return self._impl.save(*args, **kwargs)
    def recent(self, *args, **kwargs): return self._impl.recent(*args, **kwargs)
    def query(self, *args, **kwargs): return self._impl.query(*args, **kwargs)
    def load_iq(self, *args, **kwargs): return self._impl.load_iq(*args, **kwargs)
    def close(self): self._impl.close()
