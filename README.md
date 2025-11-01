# JagCorrect

## Dataset

This project uses the [GitHub Typo Corpus v1.0.0](https://github.com/mhagiwara/github-typo-corpus) for training.

To prepare the data, run the following scripts in order:
1. `python extract.py`
2. `python generate_dataset.py`

## GPU Setup

For optimal performance with TensorFlow, ensure you have the following installed:
*   **CUDA Toolkit:** 11.2
*   **cuDNN:** 8.1.1