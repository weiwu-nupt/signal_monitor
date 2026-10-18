"""SQLite result journal: one atomic packet summary plus all image predictions."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

import numpy as np

DEFAULT_DATABASE = Path(__file__).resolve().parent / 'results' / 'recognition.sqlite3'


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def model_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class RecognitionStore:
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

    def close(self):
        self.connection.close()
