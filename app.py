import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import nltk # Natural Language tool kit 
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with space
        tweet = re.sub(urlPattern," ",tweet)      
        # Replace @USERNAME to space
        tweet = re.sub(userPattern," ",tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        words = tweet.split()
        filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
        tweetwords = ''
        for word in filtered_words:
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
    return processedText


def predict_sentiment(tfidf, model, text):
    # Predict the sentiment
    textdata = tfidf.transform(preprocess(text))
    sentiment = model.predict(textdata)
    return sentiment

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form.get('tweet')
    text = [text]
    file = open('tfidf.pkl', 'rb')
    tfidf = pickle.load(file)
    file.close()
    #Load the Model
    file = open('model.pkl', 'rb')
    model = pickle.load(file)
    file.close()
    
    s = predict_sentiment(tfidf, model, text)

    if 0 in s:
        msg = 'Positive'
    else:
        msg = 'Negative'

    return render_template('index.html', prediction_text='Tweet is {}'.format(msg))

@app.route('/plots')
def plots():
    return render_template('plots1.html')

@app.route('/wordcloud')
def wordcloud():
    return render_template('wordcloud1.html')

@app.route('/confusionmatrix')
def confusionmatrix():
    return render_template('confusion1.html')

@app.route('/second')
def second():
    return render_template('second.html')



if __name__ == "__main__":
    app.run(debug=True)