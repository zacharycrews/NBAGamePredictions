import re
import nltk
import json

def clean_tweets(tweets):

    ### Cleaning Data
    # Remove mentions
    tweets = re.sub(r'[A-Za-z0-9]*@[A-Za-z_]*\.?[A-Za-z]*',' ',tweets)
    # Remove Emojis
    tweets = re.sub(r'\\\w+', ' ', tweets)
    # Remove url 
    tweets = re.sub(r'http\S+', ' ' ,tweets)
    # Remove hashtags
    tweets = re.sub(r'[@#][A-Za-z0-9_]+',' ',tweets)
    # Remove all non-word characters
    tweets = re.sub('[^A-Za-z]',' ',tweets)
    # Remove multiple space with single space
    # tweets = re.sub(' +', ' ', tweets)
    
    ### Tokenizing
    # Tokenize
    tokens = nltk.word_tokenize(tweets)
    
    # Remove Handcrafted Stopwords
    stop_words = {"d", "ll", "m", "ve", "t", "s", "re"}
    filtered = [w for w in tokens if not w.lower() in stop_words]

    # List into String
    final_tweets = ' '.join(filtered)
    return final_tweets

home_away = input("Enter 'home' or 'away': ")
train_test = input("Enter 'train' or 'test': ")

for i in range(247):
    path = 'dataset/'+home_away+'_'+train_test+'/'+home_away+'_'+train_test+str(i)+'.json'
    with open(path, 'r') as in_file:
        tweets = json.load(in_file)
    with open(path, 'w') as out_file:
        for k in tweets.keys():
            cleaned = clean_tweets(tweets[k])
            tweets[k] = cleaned
        json.dump(tweets, out_file)
