import re
import nltk


val = input("Enter 'home' or 'away': ")
for i in range(1323):
    path = 'dataset/'+val+'tweets'+val+i+'.txt'
    with open(path, 'r', encoding="utf-8") as file:
        tweets = file.read()
    with open(path, 'w+', encoding="utf-8") as file:
        file.write(clean_tweets(tweets))


def clean_tweets(tweets):

    ### Cleaning Data
    tweets = re.sub("\n", " ", tweets)
    # Remove mentions
    tweets = re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z]*',' ',tweets)
    # Remove url 
    tweets = re.sub(r'http\S+', ' ' ,tweets)
    # Remove hashtags
    tweets = re.sub(r'[@#][A-Za-z0-9_]{3,}',' ',tweets)
    # Remove all non-word characters
    tweets = re.sub('[^A-Za-z]',' ',tweets)
    # Remove multiple space with single space
    tweets = re.sub(' +', ' ', tweets)
    
    ### Tokenizing and Lemmatizing
    # Tokenize
    tokens = nltk.word_tokenize(tweets)
    # Lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for i in range(len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i])

    return tokens
