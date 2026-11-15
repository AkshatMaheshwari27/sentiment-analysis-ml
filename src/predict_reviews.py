import pandas as pd
import numpy as np
import joblib
import gensim.downloader as api

from gensim.utils import simple_preprocess


# Five custom movie reviews
reviews = [
    "This movie was absolutely fantastic and the acting was brilliant.",
    "I hated this film. The story was boring and the acting was terrible.",
    "The movie was good, but the ending was disappointing.",
    "I expected it to be terrible, but it was surprisingly enjoyable.",
    "The film was neither good nor bad. It was just average.",
    "The movie was not good at all.",
    "The acting was excellent, but the story was awful.",
    "What a wonderful waste of two hours.",
    "Some scenes were great, but overall the film was disappointing.",
    "The movie was released in 2024 and runs for two hours."
]


# Load trained TF-IDF model and vectorizer
tfidf_model = joblib.load("models/tfidf_model.joblib")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


# Load pretrained GloVe embeddings
print("Loading GloVe embeddings...")
glove = api.load("glove-wiki-gigaword-100")


# Convert a review into an average GloVe vector
def review_to_vector(review):
    words = simple_preprocess(str(review))

    word_vectors = [
        glove[word] for word in words if word in glove
    ]

    if len(word_vectors) == 0:
        return np.zeros(100)

    return np.mean(word_vectors, axis=0)


# Predict using both models
tfidf_vectors = tfidf_vectorizer.transform(reviews)
tfidf_predictions = tfidf_model.predict(tfidf_vectors)

glove_vectors = np.array([
    review_to_vector(review) for review in reviews
])

glove_model = joblib.load("models/glove_model.joblib")
glove_predictions = glove_model.predict(glove_vectors)

# Display predictions
results = []

for i, review in enumerate(reviews):
    tfidf_result = (
        "Positive" if tfidf_predictions[i] == "pos" else "Negative"
    )

    glove_result = (
        "Positive" if glove_predictions[i] == "pos" else "Negative"
    )

    results.append({
        "Review": review,
        "TF-IDF": tfidf_result,
        "GloVe": glove_result
    })

    print(f"\nReview {i + 1}: {review}")
    print("TF-IDF Prediction:", tfidf_result)
    print("GloVe Prediction:", glove_result)


# Save predictions
df = pd.DataFrame(results)
df.to_csv("outputs/custom_predictions.csv", index=False)

print("\nCustom predictions saved successfully.")
