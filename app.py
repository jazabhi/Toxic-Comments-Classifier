from flask import Flask, render_template, request, jsonify
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Load the pickled model
with open('toxiccomment.pkl', 'rb') as file:
    model = pickle.load(file)

MAX_WORDS = 200000
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_WORDS, output_sequence_length=2000, output_mode='int'
)


X = np.array([
    "This is a toxic comment.",
    "I hate this product.",
    "This is a friendly message.",
    
])

vectorizer.adapt(X)


def preprocess_text(text):
    text = np.array([text])
    vectorized_text = vectorizer(text)
    return vectorized_text


def predict_toxicity(text):
    vectorized_text = preprocess_text(text)
    prediction = model.predict(vectorized_text)
    return prediction[0]


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['user_input']
        prediction = predict_toxicity(user_input)
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        result = {label: bool(pred) for label, pred in zip(labels, prediction)}
        return render_template('index.html', result=result, user_input=user_input)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
