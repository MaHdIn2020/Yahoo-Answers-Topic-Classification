

## 📌 Yahoo Answers Topic Classification

**Machine Learning and Deep Learning Approaches for Multi-Class Text Classification**

---

## 📖 Project Overview

This project focuses on **multi-class text classification** using the **Yahoo Answers Topic Classification dataset**. The goal is to compare traditional machine learning models with deep learning architectures, including recurrent and bidirectional neural networks, to analyze their effectiveness in understanding contextual information in text.

Both **TF-IDF** and **Skip-gram (Word2Vec)** feature representations are explored, followed by extensive **manual hyperparameter tuning** and **library-based tuning** to ensure fair and meaningful comparisons.

---

## 📂 Dataset

* **Name:** Yahoo Answers Topic Classification
* **Classes:** 10 topic categories
* **Task:** Assign each question–answer pair to its correct topic
* **Text Fields Used:** Combined question title, question content, and best answer

---

## 🧹 Text Preprocessing

The following preprocessing steps were applied:

* Lowercasing text
* Removing HTML tags
* Removing punctuation and special characters
* Tokenization
* Stopword removal
* Lemmatization

These steps help reduce noise and improve feature quality for both classical and neural models.

---

## 🧠 Feature Representation

### 1️⃣ TF-IDF

Used for:

* Naive Bayes
* Logistic Regression
* Support Vector Machine
* Random Forest
* Deep Neural Network (DNN + SVD)

Dimensionality reduction was performed using **Truncated SVD** before feeding TF-IDF features into neural networks.

---

### 2️⃣ Skip-gram (Word2Vec)

* Trained using **Gensim**
* Vector size: 100
* Used for all neural network models
* Both **average Word2Vec** and **sequence-based embeddings** were explored

---

## 🤖 Models Implemented

### 🔹 Traditional Machine Learning

* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

Hyperparameters were tuned using **GridSearchCV**.

---

### 🔹 Neural Network Models

#### TF-IDF Based

* Deep Neural Network (DNN)

#### Skip-gram Based

* DNN (Average Word2Vec)
* Simple RNN
* GRU
* LSTM
* Bidirectional RNN
* Bidirectional GRU
* Bidirectional LSTM

Hyperparameters were tuned using **KerasTuner** and validation performance.

---

## ⚙️ Hyperparameter Tuning

* **GridSearchCV** used for all ML models
* **KerasTuner (RandomSearch)** used for neural networks
* Tuned parameters include:

  * Number of units
  * Dropout rate
  * Learning rate
  * Batch size
  * Number of layers

All tuning decisions were guided by **validation accuracy and Macro F1-score**.

---

## 📊 Evaluation Metrics

* **Accuracy**
* **Macro F1-score**

Macro F1-score was emphasized due to the multi-class nature of the dataset.

---

## 📈 Results Summary

* Deep learning models significantly outperform traditional ML approaches
* Skip-gram embeddings provide richer semantic representations than TF-IDF
* **Bidirectional recurrent architectures** achieve the best performance by capturing context from both past and future tokens
* Bidirectional LSTM is the top-performing model in terms of Macro F1-score

A visual comparison is shown below:

<p align="center">
  <img src="result.png" width="800">
</p>

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* TensorFlow / Keras
* KerasTuner
* Gensim
* NumPy, Pandas, Matplotlib

---

## 📁 Repository Structure

```
Yahoo-Answers-Text-Classification/
│
├── notebooks/
│   └── Yahoo_Answers_Text_Classification.ipynb
│
├── report/
│   └── Yahoo_Answers_Text_Classification_Report.pdf
│
├── figures/
│   └── result.png
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/yahoo-answers-classification.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook and run all cells.

---

## 👤 Author

**Tanjip Surait Mahdin**
Computer Science, BRAC University

---

## 📌 Notes

* This project was completed as part of an academic coursework
* Emphasis was placed on fair model comparison and proper hyperparameter tuning
* The notebook is fully reproducible


