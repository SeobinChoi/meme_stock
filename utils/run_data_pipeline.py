#!/usr/bin/env python3
"""
Meme Stock Data Processing Pipeline Runner
Simple script to run the data processing pipeline from the project root
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from data.main import main

if __name__ == "__main__":
    main() 