import pandas as pd
import numpy as np
import joblib
import gensim.downloader as api

from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/movie_reviews.csv")

X = df["text"]
y = df["label"]

# Load pretrained GloVe embeddings
print("Loading GloVe embeddings...")
glove = api.load("glove-wiki-gigaword-100")

# Convert each review into an average word vector
def review_to_vector(review):
    words = simple_preprocess(str(review))

    word_vectors = [
        glove[word] for word in words if word in glove
    ]

    if len(word_vectors) == 0:
        return np.zeros(100)

    return np.mean(word_vectors, axis=0)

print("Converting reviews into vectors...")

X_vectors = np.array([
    review_to_vector(review) for review in X
])

# Split data using the same random state as TF-IDF
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\nGloVe + Logistic Regression Results")
print("-----------------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

report = classification_report(y_test, predictions)
print("\nClassification Report:")
print(report)

# Save model
joblib.dump(model, "models/glove_model.joblib")

# Save results
with open("outputs/glove_results.txt", "w") as file:
    file.write("GloVe + Logistic Regression Results\n")
    file.write("-----------------------------------\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)

print("GloVe model and results saved successfully.")
