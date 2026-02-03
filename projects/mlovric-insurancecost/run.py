"""Small runner so the project feels easy to use.

Usage:
    python run.py eda
    python run.py train
    python run.py eval
"""

import sys
from src.main import run


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a command: eda | train | eval")
        sys.exit(1)

    command = sys.argv[1].lower()
    run(command)
