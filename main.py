# Import required libraries
import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download movie reviews dataset from nltk
nltk.download('movie_reviews')

# Prepare the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents for randomness
random.shuffle(documents)

# Convert to sentences
documents = [(" ".join(words), category) for words, category in documents]

# Extract reviews and their respective labels
reviews, labels = zip(*documents)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.3, random_state=42)

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier and train it
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Predict the labels for test data
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

# Example usage: Classifying a new review
new_review = "This movie was fantastic! The plot was thrilling and the acting was great."
new_review_vectorized = vectorizer.transform([new_review])
prediction = classifier.predict(new_review_vectorized)
print(f"The review is likely: {prediction[0]}")