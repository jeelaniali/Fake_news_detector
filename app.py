from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure form data is received correctly
        if 'news' not in request.form:
            return render_template("index.html", prediction_text="❌ Error: No text received!")

        news_text = request.form["news"]
        news_vectorized = vectorizer.transform([news_text])
        prediction = model.predict(news_vectorized)

        result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"
        return render_template("index.html", prediction_text=result, news_text=news_text)

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
