import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/movie_reviews.csv")

X = df["text"]
y = df["label"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions
predictions = model.predict(X_test_tfidf)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nTF-IDF + Logistic Regression Results")
print("------------------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model and vectorizer
joblib.dump(model, "models/tfidf_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

# Save evaluation results
with open("outputs/tfidf_results.txt", "w") as file:
    file.write("TF-IDF + Logistic Regression Results\n")
    file.write("------------------------------------\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(classification_report(y_test, predictions))

print("Model, vectorizer and evaluation results saved successfully.")
