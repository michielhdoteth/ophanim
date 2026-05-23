"""Entry point for python -m ophanim."""
import sys
import io

# Force UTF-8 encoding for Windows console (Rich uses Unicode spinner chars)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from ophanim.cli.app import app

def main():
    app()

if __name__ == "__main__":
    main()
