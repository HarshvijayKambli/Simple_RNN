 # Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

## step 2: Helper Function 
# function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

#function to process user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is the index for unknown words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500) 
    return sequence.pad_sequences([encoded_review], maxlen=500)


## step3 :Prediction function
def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

## streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive or Negative).")

user_input = st.text_area("Movie Review")

if st.button("Predict Sentiment"):
    preprocessed_input = preprocess_text(user_input)

    ## Make Prediction
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    ## display the result 
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Score: {prediction[0][0]:.4f}")
else:
    st.write("Please enter a movie review and click 'Predict Sentiment' to see the result.")    