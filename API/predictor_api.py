import pickle
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import re
from xgboost import XGBClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the models 
model = pickle.load(open("model/xgboost_VADER.p", "rb"))

def word_vectorizer(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    return [score['neg'],score['neu'],score['pos'],score['compound']]

def clean_word(text):
    # Removes different characters, symbols, numbers, some stop words
    text = re.sub('\n', '', text)
    text = text.strip().lower()
    return text

def raw_review_to_model_input(raw_input_string):
    # Converts string into cleaned text
    cleaned_text = clean_word(raw_input_string)
    return [word_vectorizer(cleaned_text)]

def predict_sentiment(raw_input_string):
    model_input = raw_review_to_model_input(raw_input_string)
    model_input = pd.DataFrame(model_input)
    results = model.predict_proba(pd.DataFrame(model_input))
    return results[0]

def make_prediction(input_chat):
    """
    Given string to classify, returns the input argument and the dictionary of 
    model classifications in a dict so that it may be passed back to the HTML page.

    Input:
    Raw string input

    Function makes sure the features are fed to the model in the same order the
    model expects them.

    Output:
    Returns (x_inputs, probs) where
      x_inputs: a list of feature values in the order they appear in the model
      probs: a list of dictionaries with keys 'name', 'prob'
    """

    if not input_chat:
        input_chat = ' '
    if len(input_chat) > 500:
        input_chat = input_chat[:500]
    pred_probs = predict_sentiment(input_chat)
    labels = ['Negative','Neutral','Positive']
    probs = [{'name': labels[i], 'prob': round(p,3)} for i,p in enumerate(pred_probs)]
    return (input_chat, probs)

if __name__ == '__main__':
    from pprint import pprint
    print("Checking to see what empty string predicts")
    print('input string is ')
    chat_in = 'bob'
    pprint(chat_in)

    x_input, probs = make_prediction(chat_in)
    print('Input values: {}'.format(x_input))
    print('Output probabilities')
    pprint(probs)