#!/usr/bin/env python3
"""
Convert a Python script into a minimal Colab-compatible notebook.

Usage:
  python scripts/py_to_ipynb.py --src colab_advanced_model_training_fixed.py --dst colab_advanced_model_training_fixed.ipynb
"""

import argparse
import json
from pathlib import Path


def to_notebook_cells(py_source: str):
    # Simple heuristic: keep the entire script in one code cell for reproducibility
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# 🚀 Leakage-Guarded Advanced Sequence Training"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": py_source.splitlines()
        }
    ]


def convert(src: Path, dst: Path):
    code = src.read_text()
    nb = {
        "cells": to_notebook_cells(code),
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.8.5",
                "mimetype": "text/x-python",
                "file_extension": ".py"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    dst.write_text(json.dumps(nb, indent=2))
    print(f"✅ Wrote notebook: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    args = parser.parse_args()

    convert(Path(args.src), Path(args.dst))


if __name__ == '__main__':
    main()


