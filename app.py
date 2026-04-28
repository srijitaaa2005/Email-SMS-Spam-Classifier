import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()



# Point NLTK to locally downloaded data (Streamlit Cloud + local safe)
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

stop_words = set(stopwords.words('english'))


def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)

  #removing special characters
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y[:]
  y.clear()

  #removing stopwords and punctuations
  for i in text:
    if i not in stop_words and i not in string.punctuation:
      y.append(i)

  text=y[:]
  y.clear()

  #apply stemming
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    #1.preprocess
    transformed_sms = transform_text(input_sms)

    #2.vectorize
    vector_input = tfidf.transform([transformed_sms])

    #3.predict
    result = model.predict(vector_input)[0]

    #4.display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not spam')