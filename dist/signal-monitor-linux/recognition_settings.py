"""Remember the user's selected checkpoint across application restarts."""
import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent / 'results' / 'recognition_settings.json'


def remembered_checkpoint(path=SETTINGS_PATH):
    try:
        saved = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(saved, dict):
            return ''
        selected = saved.get('checkpoint', '')
        return selected if isinstance(selected, str) and selected and Path(selected).is_file() else ''
    except (OSError, ValueError, TypeError):
        return ''


def remember_checkpoint(checkpoint, path=SETTINGS_PATH):
    if not Path(checkpoint).is_file():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'checkpoint': str(Path(checkpoint).resolve())}, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)
