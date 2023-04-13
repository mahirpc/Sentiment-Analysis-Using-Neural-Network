from flask import Flask, render_template, request
import pickle
import pandas as pd
import tensorflow
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import custom_object_scope

app = Flask(__name__)

# Load the trained machine learning model and other necessary files

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tockenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model=keras.models.load_model('./cnn_lstm_w2v.h5')
# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        sequence1 = tokenizer.texts_to_sequences([text])
        pad_sequence = pad_sequences(sequence1, maxlen=50)        
        sentiment = model.predict(pad_sequence,verbose=0)
        result = "Satisfied" if sentiment < 0.5 else "Not satisfied"
        colors = 'green' if sentiment < 0.5 else "red"
        d=[result,colors]
        return render_template('index.html', result=d)
    else:
        return render_template('index.html', result=[None,"white"])

if __name__ == '__main__':
    app.run(debug=True)