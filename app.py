from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import words

app = Flask(__name__)

# Load dictionary only once
english_words = set(words.words())

# Load trained model
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
ada_model = pickle.load(open("adaboost.pkl", "rb"))


# Function to detect meaningless text
def is_meaningless(text):

    text = text.lower().strip()

    # Rule 1: very short input
    if len(text) < 5:
        return True

    word_list = text.split()

    # Rule 2: repeated same word like "cheap cheap cheap"
    if len(word_list) > 1 and len(set(word_list)) == 1:
        return True

    # Rule 3: repeated characters like "thiiiiii"
    if len(set(text.replace(" ", ""))) <= 2:
        return True

    # Rule 4: dictionary check (no valid English words)
    meaningful_count = 0

    for word in word_list:
        if word in english_words:
            meaningful_count += 1

    if meaningful_count == 0:
        return True

    return False


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    review = request.form['review']

    # Detect meaningless review
    if is_meaningless(review):
        return render_template(
            "index.html",
            prediction_text="Fake Review ❌"
        )

    data = vectorizer.transform([review])

    prediction = ada_model.predict(data)[0]

    result = "Genuine Review ✅" if prediction == 1 else "Fake Review ❌"

    return render_template(
        "index.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)