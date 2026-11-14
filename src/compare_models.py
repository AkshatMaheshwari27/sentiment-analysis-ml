import matplotlib.pyplot as plt
import re

# Function to extract accuracy from a results file
def get_accuracy(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    match = re.search(r"Accuracy:\s*([\d.]+)%", content)

    if match:
        return float(match.group(1))

    raise ValueError(f"Accuracy not found in {file_path}")


# Read accuracy from saved result files
tfidf_accuracy = get_accuracy("outputs/tfidf_results.txt")
glove_accuracy = get_accuracy("outputs/glove_results.txt")

models = ["TF-IDF", "GloVe"]
accuracies = [tfidf_accuracy, glove_accuracy]


# Print comparison
print("\nModel Comparison")
print("----------------")

for model, accuracy in zip(models, accuracies):
    print(f"{model}: {accuracy}%")


# Identify the better model
best_index = accuracies.index(max(accuracies))

print("\nBest Performing Model:", models[best_index])
print("Accuracy:", accuracies[best_index], "%")


# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies, width=0.5)

plt.title("Sentiment Analysis Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

# Display accuracy above each bar
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy}%",
        ha="center"
    )

plt.tight_layout()

# Save chart
plt.savefig("outputs/model_comparison.png", dpi=300)

print("\nComparison chart saved successfully.")
