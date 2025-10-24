#!/usr/bin/env python
"""
Entry point for equipment pipeline
"""
import sys

# Add current directory to Python path
sys.path.insert(0, '.')

from pipeline import main

if __name__ == '__main__':
    sys.exit(main())
