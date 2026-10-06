"""Locate sigRec's actual training outputs without running its training scripts."""
import ast
from pathlib import Path

DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent.parent / 'sigRec'


def training_paths(script):
    """Read literal path assignments and os.path.join only; never execute source."""
    values = {}
    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if (isinstance(node, ast.Call) and not node.keywords and
            isinstance(node.func, ast.Attribute) and node.func.attr == 'join' and
            isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'path' and
            isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == 'os'):
            parts = [value(arg) for arg in node.args]
            if parts and all(isinstance(part, str) for part in parts):
                return str(Path(parts[0]).joinpath(*parts[1:]))
        return None
    try:
        tree = ast.parse(Path(script).read_text(encoding='utf-8-sig'))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                resolved = value(node.value)
                if resolved is not None:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            values[target.id] = resolved
    except (OSError, SyntaxError, ValueError):
        return {}
    return values


def discover_sigrec_checkpoint(reference_dir=DEFAULT_REFERENCE_DIR):
    # Prefer the user's requested six-class model. The five-class training
    # output remains a fallback; explicit remembered selections take priority.
    for script in ('signal_openmax.py', 'signal_classify_resnet.py'):
        path = training_paths(Path(reference_dir)/script).get('MODEL_SAVE')
        if path and Path(path).is_file():
            return str(Path(path).resolve())
    return ''
