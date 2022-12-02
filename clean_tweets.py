import re
import nltk

file = open('data.txt', 'r', encoding="utf-8")
tweets = file.read()
tweets = re.sub("\n", " ", tweets)
file.close()

def clean_tweets(tweets):

    ### Cleaning Data
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
    #print(tweets,'\n')
    return tokens

print(clean_tweets(tweets))
