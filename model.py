# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Importing essential libraries for performing NLP
import nltk # Natural Language tool kit 
import re
tweets_df = pd.read_csv('twitter1.csv')
tweets_df=tweets_df.drop(['id'], axis=1)
negative = tweets_df[tweets_df['label']==1]

count = int((tweets_df.shape[0]-negative.shape[0])/negative.shape[0])
for i in range(0, count-1):
    tweets_df = pd.concat([tweets_df, negative])

tweets_df.shape

tweets_df['length'] = tweets_df['tweet'].apply(len)

# Creating new feature word_count
tweets_df['word_count'] = tweets_df['tweet'].apply(lambda x: len(x.split()))


nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Cleaning the messages
corpus = []
wnl = WordNetLemmatizer()

for twt in list(tweets_df.tweet):
  # Cleaning special character from the tweets
  tweet = re.sub(pattern='[^a-zA-Z]', repl=' ', string=twt)

  # Converting the entire tweets into lower case
  tweet = tweet.lower()

  # Tokenizing the tweets by words
  words = tweet.split()

  # Removing the stop words
  filtered_words = [word for word in words if word not in set(stopwords.words('english'))]

  # Lemmatizing the words
  lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]

  # Joining the lemmatized words
  final_tweet = ' '.join(lemmatized_words)

  # Building a corpus of messages
  corpus.append(final_tweet)



# Creating the Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names()
# Extracting independent and dependent variables from the dataset
X = pd.DataFrame(vectors, columns=feature_names)
y = tweets_df['label']


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20)
rf.fit(X_train, y_train)

pickle.dump(rf, open('model.pkl','wb'))
pickle.dump(tfidf, open('tfidf.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
pickle.load(open("tfidf.pkl", 'rb'))  
print(model.predict(X_test))