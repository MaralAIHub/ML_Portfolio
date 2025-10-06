# IMDb Sentiment Analysis using LSTM

This repository contains a complete **Sentiment Analysis** project implemented in PyTorch, trained on the **IMDb movie reviews dataset**. The model uses a **single-layer LSTM with embeddings** to classify movie reviews as **positive** or **negative**.

![Sample Prediction](assets/sample_prediction.gif)

[Open in Google Colab](https://colab.research.google.com/github/yourusername/imdb-lstm-sentiment/blob/main/train_lstm_imdb.ipynb)

---

## 🔹 Features

* Tokenization and vocabulary building
* Padding of sequences for uniform length
* Embedding layer (randomly initialized or can use pretrained embeddings)
* Single-layer or BiLSTM for sequence modeling
* Fully connected output layer for binary classification
* Sample prediction function included
* GPU support for faster training

---

## 🔹 Dataset

* IMDb dataset (50,000 training reviews, 25,000 test reviews)
* Preprocessing:

  * Lowercasing
  * Tokenization using regex
  * Truncating long sentences for faster training
  * Optional: stopwords removal

---

## 🔹 Installation

```bash
pip install torch torchvision torchaudio torchtext datasets numpy
```

---

## 🔹 Usage

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/imdb-lstm-sentiment.git
cd imdb-lstm-sentiment
```

2. **Run the notebook or Python script**

   * Make sure GPU is enabled in Colab or your local environment for faster training.

```bash
python train_lstm_imdb.py
```

3. **Sample predictions**

```python
predict("This movie was amazing!")   # Expected: Positive
predict("I hated this movie, it was so boring.")  # Expected: Negative
```

---

## 🔹 Code Explanation

* **tokenize(text)**: splits text into words.
* **Vocabulary building**: maps words to indices for embedding.
* **Embedding layer**: converts word indices to dense vectors.
* **LSTM/BiLSTM layer**: captures sequential dependencies.
* **Fully Connected layer**: maps LSTM output to class logits.
* **Softmax & Loss function**: converts logits to probabilities and computes error.
* **Training loop**: updates weights based on CrossEntropyLoss.
* **Prediction function**: outputs probability for positive/negative sentiment.

---

## 🔹 Improving Accuracy

* Use **pretrained embeddings** (GloVe, FastText)
* Increase **hidden size** or **number of LSTM layers**
* Use **BiLSTM** or **attention mechanisms**
* Increase **number of epochs** for better convergence

---

## 🔹 License

This project is open-source and free to use.
