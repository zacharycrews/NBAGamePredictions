import os
import re
import nltk

def remove_emojis(tweets):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweets)


def clean_tweets(tweets):
    # Remove emails, urls, mentions, hashtags, and other irrelevant chars
    tweets = re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z]*',' ',tweets)
    tweets = re.sub(r'https:\/\/t\.co\/[A-Za-z0-9]*',' ',tweets)
    tweets = re.sub(r'[@#][A-Za-z0-9_]{3,}',' ',tweets)
    tweets = re.sub('[^A-Za-z\s]',' ',tweets)

    #tweets = tweets.replace("\n"," ")
    
    # Remove emojis
    #tweets = remove_emojis(tweets)
    
    # Tokenize
    tokens = nltk.word_tokenize(tweets)

    # Lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for i in range(len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i])
    
    return tokens

file = open('dataset/hometweets/home6.txt', 'r')
tweets = file.read()
file.close()
print(clean_tweets(tweets))
