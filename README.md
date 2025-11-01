# JagCorrect

This project aims to correct common typing errors using a deep learning model.

## Setup Guide for Non-Experienced Users

This guide will walk you through setting up the project environment and running the scripts. We will use `conda` for managing our project's dependencies, which helps avoid conflicts with other Python projects on your system.

### Step 1: Install Miniconda or Anaconda

If you don't have `conda` installed, you'll need to install either [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for a minimal installation) or [Anaconda](https://www.anaconda.com/products/individual). Follow the instructions on their respective websites for your operating system.

### Step 2: Create and Activate the Conda Environment

This project requires a specific Python version (3.8) and TensorFlow 2.10.0, which is compatible with CUDA 11.2 and cuDNN 8.1.1 for GPU acceleration.

1.  **Open your terminal or Anaconda Prompt.**
2.  **Create the `tf210_py38` conda environment:**
    ```bash
    conda create -n tf210_py38 python=3.8
    ```
    When prompted to proceed, type `y` and press Enter.
3.  **Activate the newly created environment:**
    ```bash
    conda activate tf210_py38
    ```
    You should see `(tf210_py38)` at the beginning of your terminal prompt, indicating that the environment is active.

### Step 3: Install Project Dependencies

With your `tf210_py38` environment activated, you can now install the necessary libraries for the project.

1.  **Navigate to the project directory** (where this `README.md` file is located) in your terminal.
2.  **Install the dependencies using pip:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install all the required Python packages.

### Step 4: Prepare the Dataset

This project uses the [GitHub Typo Corpus v1.0.0](https://github.com/mhagiwara/github-typo-corpus) for training.

To prepare the data, run the following scripts in order *within your activated `tf210_py38` environment*:

1.  **Extract data:**
    ```bash
    python extract.py
    ```
2.  **Generate dataset:**
    ```bash
    python generate_dataset.py
    ```

### Step 5: Run a Script

To run any of the project's scripts, ensure your `tf210_py38` Conda environment is active.

*   **To start the main AI training script:**
    ```bash
    start.bat
    ```
    This script is configured to use the Python executable from your `tf210_py38` environment.

*   **To run other Python scripts (e.g., tests):**
    ```bash
    python test_ai.py
    ```
    Make sure you are in the project's root directory and the `tf210_py38` environment is active.

### GPU Setup (Optional, for optimal performance)

For optimal performance with TensorFlow, especially for training, ensure you have the following installed on your system:

*   **CUDA Toolkit:** Version 11.2
*   **cuDNN:** Version 8.1.1

TensorFlow 2.10.0 is compatible with these versions. If you have a compatible NVIDIA GPU, installing these will allow TensorFlow to utilize it for faster computations. Refer to the official TensorFlow documentation for detailed installation instructions for GPU support.

### Verification (Recommended)

To ensure you are using the correct Python environment:

1.  **After activating your `tf210_py38` environment**, run:
    ```bash
    python --version
    ```
    It should output `Python 3.8.x`.
2.  **To see which Python executable is being used**, run:
    ```bash
    where python
    ```
    (on Windows) or
    ```bash
    which python
    ```
    (on Linux/macOS). The output should point to the `python.exe` within your `tf210_py38` conda environment (e.g., `C:\Users\Admin\miniconda3\envs\tf210_py38\python.exe`).
