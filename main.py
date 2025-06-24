# main.py
import pandas as pd
import numpy as np
import string
import nltk
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load data
fake = pd.read_csv('dataset/Fake.csv')
true = pd.read_csv('dataset/True.csv')

# Add labels
fake['label'] = 1  # Fake
true['label'] = 0  # Real

# Combine datasets
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

data['text'] = data['text'].apply(clean_text)

# Remove stopwords
stop_words = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

# Split data
X = data['text']
y = data['label']

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model & vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")