import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from pprint import pprint
import streamlit as st
from random import randint

model = joblib.load("model.sav")
vect = joblib.load("vectorizer.sav")
trans = joblib.load("transformer.sav")
enc = joblib.load("encoder.sav")

def predict(corpus):
    x = vect.transform(corpus)
    x = trans.transform(x)
    result = model.predict(x)
    result = enc.inverse_transform(result)
    result = "Sentiment: " + str(result)
    return result


st.title("Ravish Jha's Text Sentiment Prediction Project")
st.markdown("This project predicts sentiment of the input text using a trained machine learning model")
st.divider()
with st.container():
    corpus = st.text_area("Input your text")
    with st.container():
        col1 , col2 = st.columns(2)
        with col1:
            gen = st.button("Predict Result")
        with col2:
            load = st.button("Load Example")

corpus = [corpus]
examples = ["The quirky script is packed with good lines and off-the-wall moments and the stories are surprisingly moving in places." , 
            "Oh it's fun. An old-fashioned, good-hearted action film with a sense of humor." , 
            "Small pleasures aside, the movie doesn't offer anything particularly memorable or inventive." , 
            "With a cast that reads like the Vogue Oscar party guest list, Valentine's Day should have been can't-miss cinema instead of standard Hollywood schmaltz."]
try:
    if load:
        corpus = st.text_area("Input your text" , examples[(randint(0,3))])
        st.warning("Click on 'Predict Result'")
        if gen:
            st.success(predict(corpus))
    else:
        if len(corpus) == 0:
            st.error("Please enter some text")
        else:
            if gen:
                st.success(predict(corpus))
except:
    st.error("You seem to have discovered a bruh moment. Please report it with a screen capture at ravisjha.2002@yahoo.com")

