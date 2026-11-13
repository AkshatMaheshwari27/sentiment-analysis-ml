import nltk
import pandas as pd

# Load the locally downloaded dataset
nltk.data.path.append("data/nltk_data")

from nltk.corpus import movie_reviews

reviews = []

# Read all reviews and their labels
for file_id in movie_reviews.fileids():
    text = movie_reviews.raw(file_id)
    label = movie_reviews.categories(file_id)[0]

    reviews.append({
        "text": text,
        "label": label
    })

# Convert into a DataFrame
df = pd.DataFrame(reviews)

# Save dataset as CSV
df.to_csv("data/movie_reviews.csv", index=False)

print("Dataset saved successfully")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())
