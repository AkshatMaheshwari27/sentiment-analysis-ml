# Movie Review Sentiment Analysis

A machine learning project that classifies movie reviews as Positive or Negative using two approaches: TF-IDF with Logistic Regression and pretrained GloVe word embeddings with Logistic Regression.

## 1. Project Overview

The objective of this project is to understand how machine learning models process text and identify sentiment.

We train two models using the same dataset and compare their performance on an unseen test set.

## 2. Dataset

- **Dataset:** NLTK Movie Reviews
- **Total reviews:** 2,000
- **Positive reviews:** 1,000
- **Negative reviews:** 1,000
- **Training samples:** 1,600
- **Testing samples:** 400

The processed dataset is stored in `data/movie_reviews.csv`.

## 3. Models Implemented

### Model 1: TF-IDF + Logistic Regression

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features based on the importance of words.

Configuration:
- Maximum features: 10,000
- N-grams: Unigrams and bigrams
- Minimum document frequency: 2
- Classifier: Logistic Regression

**Test accuracy: 81.75%**

### Model 2: GloVe + Logistic Regression

GloVe (Global Vectors for Word Representation) uses pretrained word embeddings to represent words as numerical vectors.

In this implementation:
- Pretrained embeddings: `glove-wiki-gigaword-100`
- Embedding dimensions: 100
- Each review is represented by averaging its word vectors.
- Classifier: Logistic Regression

**Test accuracy: 73.75%**

## 4. Model Comparison

Both models were evaluated using the same 80:20 train-test split.

| Model | Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | 81.75% |
| GloVe + Logistic Regression | 73.75% |

TF-IDF performed better in this experiment, with an 8 percentage-point advantage.

The comparison chart is available at `outputs/model_comparison.png`.

## 5. Custom Testing

Both trained models were tested on 10 manually written reviews.

The test cases included:
- Clearly positive and negative reviews
- Negation
- Mixed sentiment
- Sarcasm
- Neutral and factual statements

The predictions are saved in `outputs/custom_predictions.csv`.

These tests demonstrate that both models can struggle with negation, sarcasm, mixed opinions and neutral statements.

## 6. Project Structure

```text
week3-sentiment/
├── data/
│   └── movie_reviews.csv
├── models/
│   ├── glove_model.joblib
│   ├── tfidf_model.joblib
│   └── tfidf_vectorizer.joblib
├── outputs/
│   ├── custom_predictions.csv
│   ├── glove_results.txt
│   ├── model_comparison.png
│   └── tfidf_results.txt
├── src/
│   ├── load_data.py
│   ├── train_tfidf.py
│   ├── train_glove.py
│   ├── compare_models.py
│   └── predict_reviews.py
└── README.md
```

## 7. Installation and Execution

### Requirements

- Python 3.13
- pip
- Git

### Install dependencies

```bash
pip install -r requirements.txt
```

### Prepare the dataset

```bash
python src/load_data.py
```

### Train the TF-IDF model

```bash
python src/train_tfidf.py
```

### Train the GloVe model

```bash
python src/train_glove.py
```

### Compare the models

```bash
python src/compare_models.py
```

### Run custom predictions

```bash
python src/predict_reviews.py
```

Note: The GloVe model is downloaded automatically through Gensim on its first use. Internet access is required for this initial download. The NLTK movie reviews corpus must also be available locally when preparing the dataset.

## 8. Limitations

- The dataset contains only positive and negative labels.
- Neutral sentiment is not explicitly supported.
- Averaging GloVe vectors loses word order and much of the contextual information.
- Sarcasm, negation and mixed sentiment can lead to incorrect predictions.
- The results are specific to this dataset, train-test split and model configuration.

## 9. Future Improvements

- Experiment with TF-IDF-weighted GloVe embeddings.
- Explore Word2Vec and other embedding techniques.
- Test sequence-based models.
- Explore Transformer models and self-attention.
- Improve handling of negation, sarcasm and neutral sentiment.

## 10. Technologies Used

- Python
- Pandas and NumPy
- NLTK
- Scikit-learn
- Gensim
- Matplotlib
- Joblib