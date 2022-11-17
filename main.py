###################################################################################
#
#  tweepy version 3.7.0
#  bot.py and last_seen.txt must be in the same folder
#  change last_seen.txt to the tweet id b4 the one you want to respond to
#  ie if u want to respnd to tweet 4 make sure last_seen.txt is tweet 3's id
#  last_seen.txt = 1587982753596731392
#
####################################################################################

import tweepy
import time
import requests
import json
import coremltools as ct

# naive bayes model
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

model = ct.models.MLModel('NLPTWEET.mlmodel')

# training naive bayed model
df = pd.read_csv('CleanDataSet.csv', encoding='latin-1')
df.isnull()
df.isnull().sum().sum()
df.dropna(inplace=True)
df.describe()
df = df.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
df.fillna(0, inplace=True)
# print(df.info())
# print(df.head())
df['bot_or_human'] = df['bot_or_human'].map({'bot': 0, 'human': 1})
X = df['text']
y = df['bot_or_human']
global cv
cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Naive Bayes Classifier\
global clf
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))


# api keys
consumer_key = 'K8dLX4dt0MmIRCH1m41nOzr2W'
consumer_secret = '1sMbwb8ltjHruk2nfE1qRRMsfUH7mi17j1C4U3feXrfx2OEGpe'
key = '1582906439101607937-7gE11jkxeNaQxL8THUsf7lmV9yTenG'
secret = 'YPXRQdG0Js3wrzTXgCvZnXvDHGfN4geJ8xPreZKyhlLtL'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMd4iQEAAAAAI0iMpsZHXkATAG6cpcDw3PWRXD4%3DCP8AvV4iVS4xwMHCBz7akyach8Kqcl9g0wPCMuw6kNI6WTjgqV'

# api authorization
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(key, secret)
api = tweepy.API(auth)

# last seen twitter id, to avoid duplicate replies
FILE_NAME = 'last_seen.txt'


# functions to read and write to the last seen id file
def read_last_seen(FILE_NAME):
    file_read = open(FILE_NAME, 'r')
    last_seen_id = int(file_read.read().strip())
    file_read.close()
    return last_seen_id


def store_last_seen(FILE_NAME, last_seen_id):
    file_write = open(FILE_NAME, 'w')
    file_write.write(str(last_seen_id))
    file_write.close()
    return


# reply function
def reply():
    counter = 0
    for tweet in reversed(tweets):
        # only replies to tweets with the #ultimatebot in it
        if '#ultimatebot' in tweet.full_text.lower():

            convo_id = analyze_tweet()
            # need to change this to the tweet of the conversation ID
            url = 'https://api.twitter.com/2/tweets/{}'.format(convo_id)

            tweetNLP = getConvoTweet(url)
            # print(tweetNLP)

            #####

            ##df_new = pd.DataFrame([tweetNLP], columns=['text'])
            ##X = df_new['text']

            # print('X is' + str(type(X)))

            # put the tweet into panda data frame
            ##A = cv.transform(X)
            # print("A is " + str(type(A)))

            ##prediction = clf.predict(A)
            # print(B)
            # [0] is bot [1] is human
            print(f'The tweet predicted is: {tweetNLP} ')
            print(type(tweetNLP))
            prediction = model.predict({'text': str(tweetNLP)})
            print(f'precidction output is {prediction}')
            print(type(prediction))
            newpredict = prediction.get('label')
            if (str(newpredict) == '1'):
                result = 'human'

            if (str(newpredict) == '0'):
                result = 'bot'
            # send the reply (reply message, tweet id)

            api.update_status("@" + tweet.user.screen_name + " This tweet was posted by a " + result, tweet.id)

            # store the new tweet id
            store_last_seen(FILE_NAME, tweet.id)


def create_url():
    id = read_last_seen(FILE_NAME)
    url = 'https://api.twitter.com/2/tweets?ids={}&tweet.fields=conversation_id'.format(id)
    return url


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def getConvoTweet(convoUrl):
    response = connect_to_endpoint(convoUrl)
    x = json.loads(json.dumps(response))
    y = x.get('data')

    return (y.get('text'))


# use this function to get the tweet to run through the model
def analyze_tweet():
    url = create_url()
    json_response = connect_to_endpoint(url)
    # extracts conversation_id from json file
    x = json.loads(json.dumps(json_response))
    y = x.get('data')
    return (y[0]['conversation_id'])


# main
while True:
    # calls on all timeline mentions (when someone @'s the bot)
    tweets = api.mentions_timeline(read_last_seen(FILE_NAME), tweet_mode='extended')

    # the reply function

    # print(convo_id)
    reply()

    # wait timer b4 next reply
    time.sleep(15)
