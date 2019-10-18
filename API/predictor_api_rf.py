import pickle
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spacy.lang.en import English
import gensim
import string

# Load the models 
model = pickle.load(open("model//doc2vec/paragraphs/RF.p", "rb"))
embedding = gensim.models.Word2Vec.load("model/doc2vec/paragraphs/paragraphsReviews.d2v")

       
class word_vectorizer(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.parser = English()
        self.punctuations = string.punctuation

    def tokenizer(self,X):
        # Lemmatizes the reviews
            reviews_tokens = []
            for text in X:
                parsedText = self.parser(text)
                token_list = []
                for token in parsedText:
                    if token.lemma_ != "-PRON-":
                        if (token.is_stop == False) & (token.lemma_ not in self.punctuations):
                            token_list.append(token.lemma_)
                reviews_tokens.append(token_list)
            return reviews_tokens

    def vectorizer(self,X):
        # Projects the reviews to the vector space using the doc2vec embedding
            vecs = []
            for review in X:
                revVec = []
                for w in review:
                    # If the word is in the vocabulary, return the learned projection
                    if w in embedding.wv.vocab.keys():
                        wvec = embedding.wv.get_vector(w)
                        revVec.append([wvec])
                    # If the word is new, infer its vector
                    else:
                        wvec = embedding.infer_vector([w])
                        revVec.append([wvec])
                if revVec == []:
                    return []
                meanVector = np.mean(revVec, axis=0).reshape((300,))
                vecs.append(meanVector)
            return vecs

    def vectorize(self):
        self.sentence = self.tokenizer(self.sentence)
        self.sentence = self.vectorizer(self.sentence)
        return self.sentence



def clean_word(text):
    # Removes different characters, symbols, numbers, some stop words
    text = re.sub('\n', '', text)
    text = text.strip().lower()
    return [text]

def raw_review_to_model_input(raw_input_string):
    # Converts string into cleaned text
    cleaned_text = clean_word(raw_input_string)
    wv = word_vectorizer(cleaned_text)
    vector = wv.vectorize()
    return vector

def predict_sentiment(raw_input_string):
    model_input = raw_review_to_model_input(raw_input_string)
    if model_input == []:
        return [0,1,0]
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
    chat_in = ''
    pprint(chat_in)

    x_input, probs = make_prediction(chat_in)
    print('Input values: {}'.format(x_input))
    print('Output probabilities')
    pprint(probs)