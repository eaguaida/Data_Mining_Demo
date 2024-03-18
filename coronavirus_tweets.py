import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
nltk.download('punkt')
nltk.download('stopwords')

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
    df = pd.read_csv(data_file, encoding='latin1')
    return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    sentiment_counts = df['Sentiment'].value_counts()
    return sentiment_counts

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(sentiment_counts):
    second_pop_sentiment = sentiment_counts.index[1]
    return second_pop_sentiment

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    df['TweetAt'] = pd.to_datetime(df['TweetAt'])
    extremely_positive_dates = df[df['Sentiment'] == 'Extremely Positive']['TweetAt'].dt.date.value_counts()
    date_most_extremely_positive = extremely_positive_dates.idxmax()
    return str(date_most_extremely_positive)

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
    df['ProcessedTweet'] = df['OriginalTweet'].str.lower()
    return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['ProcessedTweet'] = df['ProcessedTweet'].str.replace('[^a-z\s]', ' ', regex=True)
    return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df['ProcessedTweet'] = df['ProcessedTweet'].apply(lambda x: re.sub('\s+', ' ', x).strip())
    return df

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    df['Tokens'] = df['ProcessedTweet'].apply(word_tokenize)
    return df

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    total_word_count = sum(tdf['Tokens'].apply(len))
    return total_word_count

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    distinct_words = set(word for tweet in tdf['Tokens'] for word in tweet)
    return len(distinct_words)

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
    all_words = [word for tokens in tdf['Tokens'] for word in tokens]
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(k)
    return [word for word, count in most_common_words]

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    stop_words = set(stopwords.words('english'))
    tdf['Tokens'] = tdf['Tokens'].apply(
        lambda tokens: [word for word in tokens if word not in stop_words and len(word) > 2]
    )
    return tdf

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    stemmer = PorterStemmer()
    tdf['Tokens'] = tdf['Tokens'].apply(
        lambda tokens: [stemmer.stem(word) for word in tokens]
    )
    return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
    # Vectorize the tweets with the best parameters
    vectorizer = CountVectorizer(max_df=0.5, min_df=2, max_features=10000)
    X = vectorizer.fit_transform(df['ProcessedTweet'])
    
    # Fit a Multinomial Naive Bayes classifier with the best alpha value
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X_train, y_train)
    
    # Predict sentiments for the test set
    y_pred = classifier.predict(X_test)
    
    return y_pred, y_test

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    return round(accuracy, 3)