import pandas as pd
import string
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE

# ✅ Ensure stopwords are available
nltk.download('stopwords')

# ✅ Load Fake and Real News datasets
try:
    df_fake = pd.read_csv("Fake.csv")  # Load Fake News
    df_real = pd.read_csv("True.csv")  # Load Real News
    print("\n✅ Datasets Loaded Successfully!")
except FileNotFoundError:
    print("\n❌ Error: 'Fake.csv' or 'True.csv' not found! Make sure the files are in the correct folder.")
    exit()

# ✅ Add labels: Fake = 0, Real = 1
df_fake["label"] = 0
df_real["label"] = 1

# ✅ Select only required columns (text & label)
df_fake = df_fake[["text", "label"]]
df_real = df_real[["text", "label"]]

# ✅ Merge the datasets
df = pd.concat([df_fake, df_real]).reset_index(drop=True)

# ✅ Ensure 'text' has no missing values
df['text'] = df['text'].fillna("").astype(str)

# ✅ Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenize by splitting on spaces
    stop_words = set(stopwords.words('english'))  # Load stopwords
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# ✅ Apply text cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# ✅ Check label distribution before training
print("\n🔹 Final Dataset Label Distribution (Real vs Fake):")
print(df['label'].value_counts())

# ✅ Convert text into numerical features (TF-IDF with optimized parameters)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", min_df=5, max_df=0.9)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# ✅ Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ✅ Split dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# ✅ Print label distribution in training and testing sets
print("\n🔹 Training Data Label Distribution:")
print(pd.Series(y_train).value_counts())

print("\n🔹 Testing Data Label Distribution:")
print(pd.Series(y_test).value_counts())

# ✅ Train an Ensemble Model (Random Forest + Gradient Boosting)
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# ✅ Ensure RandomForest is fitted separately before extracting feature importance
rf_model.fit(X_train, y_train)  

# ✅ Use Voting Classifier for Better Accuracy
model = VotingClassifier(estimators=[("rf", rf_model), ("gb", gb_model)], voting="soft")
model.fit(X_train, y_train)

# ✅ Print top 20 most important words (Fix for Voting Classifier)
feature_names = vectorizer.get_feature_names_out()
top_words = sorted(zip(rf_model.feature_importances_, feature_names), reverse=True)[:20]
print("\n🔹 Top 20 Important Words for Fake News Detection:")
for coef, word in top_words:
    print(f"{word}: {coef:.4f}")

# ✅ Evaluate the model
y_pred = model.predict(X_test)
print("\n🔹 Model Performance 🔹")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ Save the model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n✅ Model training complete! Saved as 'fake_news_model.pkl'.")

# ✅ Function to predict news authenticity
def predict_news(news_text):
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    news_vector = vectorizer.transform([news_text])
    prediction_proba = model.predict_proba(news_vector)  # Get probability scores
    prediction = model.predict(news_vector)[0]

    real_prob = prediction_proba[0][1]  # Probability of real news
    fake_prob = prediction_proba[0][0]  # Probability of fake news

    print(f"\n🔹 Prediction Confidence: Real: {real_prob:.4f}, Fake: {fake_prob:.4f}")

    return "🟢 Real News" if prediction == 1 else "🔴 Fake News"

# ✅ Test the model with a sample news article
sample_news = "Jones certified U.S. Senate winner despite Moore challenge!"
print("\n🔹 Testing the model...")
print(f"News: {sample_news}")
print(f"Prediction: {predict_news(sample_news)}")
