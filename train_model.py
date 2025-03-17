import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords if not available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to clean text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load the datasets
true_df = pd.read_csv("true.csv")
fake_df = pd.read_csv("fake.csv")

# Assign labels: 1 for Real News, 0 for Fake News
true_df["label"] = 1
fake_df["label"] = 0

# Keep only necessary columns
true_df = true_df[["text", "label"]]
fake_df = fake_df[["text", "label"]]

# Combine datasets
df = pd.concat([true_df, fake_df], axis=0).dropna()

# Apply text preprocessing
df["text"] = df["text"].apply(preprocess_text)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print("\n🔹 Model Performance 🔹")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\n✅ Model and vectorizer saved successfully!")
