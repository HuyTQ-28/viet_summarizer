# Vietnamese Text Summarization

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

## Overview

An end-to-end project for abstractive text summarization of Vietnamese news articles, featuring a Transformer model built from scratch with PyTorch and a web interface powered by FastAPI.

## ✨ Features

- **Transformer from Scratch:** The core summarization model is a standard Encoder-Decoder Transformer, implemented from the ground up using PyTorch.
- **Beam Search Inference:** Utilizes beam search decoding to generate higher-quality, more coherent summaries.
- **Web Interface:** A user-friendly web application built with FastAPI and Jinja2 templates for easy interaction and demonstration.
- **Training and Evaluation:** Complete scripts for training the model from scratch and evaluating its performance.
- **Data Processing:** Includes utilities for cleaning and processing Vietnamese text for the summarization task.

## 🌟 Demo

The web interface allows users to input text and receive a summary.

_(Note: The image link below should be replaced with a public URL, for example, by uploading the image to a GitHub issue as described previously.)_
![Application Demo](checkpoints/Screenshot%202025-08-23%20221900.png)

## 🛠️ Technology Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **ML/DL:** PyTorch, SentencePiece
- **Data Handling:** Pandas
- **Containerization:** Docker

## 📂 Project Structure

```
viet_summarizer/
├── app.py                  # FastAPI application for the web interface
├── checkpoints/            # Stores trained model checkpoints
├── data/                   # Contains processed data (CSV files)
├── model/                  # Defines the Transformer model architecture
│   ├── attention.py        #   - Multi-Head Attention layer
│   ├── encoder.py          #   - Encoder block
│   ├── decoder.py          #   - Decoder block
│   ├── layers.py           #   - Helper layers (e.g., Feed-Forward, Normalization)
│   └── transformer.py      #   - Main Transformer model assembly
├── static/                 # Static files (CSS, JS) for the frontend
├── templates/              # HTML templates for the web UI
├── tokenizers/             # Stores the trained SentencePiece tokenizer
├── utils/                  # Helper functions (config, data loading, processing)
├── Dockerfile              # Configuration for building the Docker container
├── data_exploration.ipynb  # Notebook for data exploration and analysis
├── inference.py            # Script to run inference on a single text
├── modal_train.py          # Script for distributed training on Modal
├── requirements.txt        # List of required Python libraries
├── run_evaluation.py       # Script to evaluate model performance
├── test.ipynb              # Notebook for testing functionalities
└── train.py                # Script to train the model
```

## Model Performance

### Dataset

- The model was trained on the **VietNews-Abs-Sum** dataset, a collection of articles for the Vietnamese Abstractive Summarization task. This dataset is derived from the original Vietnews (VNDS) dataset, with articles collected from major Vietnamese online newspapers such as `tuoitre.vn`, `vnexpress.net`, and `nguoiduatin.vn`.
  **Source:** [ithieund/VietNews-Abs-Sum on Hugging Face](https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum)
- Performance:
  ROUGE-1 F1 : 43.17%
  ROUGE-2 F1 : 14.36%
  ROUGE-L F1 : 28.86%

## ⚙️ Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) (optional, for dependency management) or pip

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-link>
    cd viet_summarizer
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

> **Important Note:** Due to distribution constraints, the pre-trained model checkpoint is not provided in this repository. You must train the model yourself by following Step 1 before you can run the web application or inference scripts.

### 1. Train the Model

To train the model from scratch, ensure your dataset is correctly placed in the `data/processed` directory and run:

```bash
python train.py
```

_Note: Training requires a GPU and can take a significant amount of time_

After the training process is complete, the best model checkpoint will be saved as `checkpoints/transformer_summarizer_best.pt`. This file is required for the following steps.

### 2. Run the Web Application

Once the model checkpoint is available, you can start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Navigate to `http://localhost:8000` in your browser.

### 3. Command-Line Inference

To summarize a piece of text directly from the terminal:

```bash
python inference.py --text "Your long Vietnamese text to be summarized..."
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 🤝 Acknowledgements

- The dataset used in this project is based on the work of the [Vietnews (VNDS) authors](https://github.com/ThanhChinhBK/vietnews).
- Special thanks to the Hugging Face team for hosting the [VietNews-Abs-Sum dataset](https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum).
