# JagCorrect

1.  **Create and activate a new Conda environment:**
    ```bash
    conda create -n tf210_py38 python=3.8
    conda activate tf210_py38
    ```

2.  **Install dependencies:**
    This project was developed using Python 3.8 and TensorFlow 2.10.0, which is compatible with CUDA 11.2 and cuDNN 8.1.1. For GPU support, ensure these are installed and configured correctly.

    Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the dataset** by following the instructions in the Dataset section below.

4.  **Run a script:** Ensure your Conda environment is active before running any scripts (e.g., `start.bat`, `python test_ai.py`).

## Dataset

This project uses the [GitHub Typo Corpus v1.0.0](https://github.com/mhagiwara/github-typo-corpus) for training.

To prepare the data, run the following scripts in order:
1. `python extract.py`
2. `python generate_dataset.py`

## GPU Setup

For optimal performance with TensorFlow, ensure you have the following installed:
*   **CUDA Toolkit:** 11.2
*   **cuDNN:** 8.1.1
