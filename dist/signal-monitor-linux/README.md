# Signal Monitor for Linux

This package contains application source only. It deliberately excludes
`data/`, recognition records, model weights, Python, and third-party libraries.

On the target Linux computer, where Python 3, PyQt6, NumPy, and pyqtgraph are
already installed, run:

```sh
chmod +x run.sh
./run.sh
```

Or run `python3 main.py` directly. Select external IQ files and model weights
from the application when needed.