"""(Deprecated) Transformer training.

This repository's **final path** is the linear + TF-IDF pipeline with Optuna HPO.

We keep this filename only to avoid breaking old references, but the transformer
approach is intentionally removed to keep the Colab environment lightweight
(no torch/transformers dependency).

Use instead:
    - call-center/src/optuna_search.py
    - call-center/src/train_linear.py
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "[train_transformer] This script is deprecated/disabled. "
        "Use optuna_search.py (linear TF-IDF) instead."
    )
    sys.exit(2)


if __name__ == "__main__":
    main()



