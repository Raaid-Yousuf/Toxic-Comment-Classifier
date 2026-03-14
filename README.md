# 🧠 Toxic Comments Classifier

A **Deep Learning based Multi-Label Toxic Comment Classification System** that detects different types of toxic behavior in online comments. This project uses **Natural Language Processing (NLP)** and **TensorFlow/Keras** to classify comments into multiple toxicity categories such as insults, threats, and identity hate.

Online discussions often become hostile due to abusive language. The goal of this project is to build a model that can automatically detect toxic content and help platforms maintain **healthier and more respectful conversations**.

## 📌 Project Overview

Discussing important topics online can sometimes lead to harassment and abusive behavior. Many platforms struggle to moderate discussions effectively.

This project focuses on building a **multi-headed classification model** capable of detecting different forms of toxic comments.

The model predicts the probability of the following toxicity categories:

- Toxic

- Severe Toxic

- Obscene

- Threat

- Insult

- Identity Hate

The dataset used consists of **Wikipedia talk page comments** labeled by human annotators.

## 📂 Dataset

The dataset used in this project is from the **Toxic Comment Classification Challenge.** It contains Wikipedia comments labeled for different types of toxic behavior.

#### 🔗 Dataset Link:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

## 📂 Dataset Description

The dataset contains thousands of Wikipedia comments labeled for toxic behavior.

Each comment may belong to **multiple categories simultaneously**, making this a **multi-label classification problem.**

#### Target Labels
**Label	Description**
+ toxic::          General toxic or rude comment
+ severe_toxic::	 Extremely toxic content
+ obscene::	       Obscene or vulgar language
+ threat::	       Threatening language
+ insult::	       Personal insults
+ identity_hate::	 Hate speech targeting identity groups

#### ⚠️ Disclaimer:
The dataset contains offensive language which may be disturbing to some readers.

## 🧰 Tech Stack
**`Programming Language`**

+ Python

**`Libraries & Frameworks`**

+ TensorFlow / Keras

+ Pandas

+ NumPy

+ Matplotlib

+ Gradio

**`NLP Components`**

+ Text Vectorization

+ Tokenization

+ Embedding Layer

+ Bidirectional LSTM

## 🏗 Model Architecture

The project implements a **Deep Learning model using Bidirectional LSTM** to capture contextual relationships in text.

`Architecture Overview`
Input Layer

     ↓
   
Text Vectorization

     ↓
   
Embedding Layer

     ↓
   
Bidirectional LSTM

     ↓
   
Dense Layer (ReLU)

     ↓
   
Dense Layer (ReLU)

     ↓
   
Dense Layer (ReLU)
 
     ↓
   
Output Layer (6 units - Sigmoid)

`Output`

The model outputs **six probability scores**, one for each toxicity class.

Because multiple labels can be present in a single comment, sigmoid activation is used instead of softmax.

## ⚙️ Project Workflow

The notebook follows these main steps:

**1️⃣ Data Preprocessing**

1. Cleaning text

2. Tokenization

3. Text Vectorization

4. Padding sequences

5. Preparing multi-label targets

**2️⃣ Model Creation**

1. Embedding layer for word representation

2. Bidirectional LSTM for contextual understanding

3. Dense layers for classification

**3️⃣ Model Training**

1. Training on labeled Wikipedia comments

2. Optimizing using binary cross-entropy loss

**4️⃣ Model Evaluation**

1. Precision

2. Recall

3. Accuracy metrics

**5️⃣ Predictions**

1. The model predicts toxicity probabilities for unseen comments.

**6️⃣ Interactive Interface**

1. A Gradio interface is implemented to allow users to test the model by entering their own comments.

## 📈 Future Improvements

Possible enhancements for the project:

- Fine-tuning transformer models (BERT / RoBERTa)

- Larger and more diverse datasets

- Better handling of sarcasm and context

- Real-time moderation systems

- Deployment using APIs

## 🤝 Contributing

Contributions are welcome!
If you'd like to improve the project, feel free to submit a pull request.

## 📜 License

This project is for **educational and research purposes.**
