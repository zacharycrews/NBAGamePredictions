import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ReadCSV import training_games, training_labels, testing_games, testing_labels
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#####                                                                               TRAINING SECTION


### FORMAT OF SENTIMENT_FEATURES:
# [blob_home_pos, blob_home_neg, blob_away_pos, blob_away_neg, vader_home_pos, vader_home_neg, vader_away_pos, vader_away_neg]
sentiment_features = [[0,0,0,0,0,0,0,0] for i in range(len(training_games))]

### READ IN DATA FROM JSON
def ReadInFile(index):
    home_path = "dataset/home_train/home_train"+str(index)+".json"
    home_file = open(home_path)
    home_tweets = json.load(home_file)
    home_file.close()

    away_path = "dataset/away_train/away_train"+str(index)+".json"
    away_file = open(away_path)
    away_tweets = json.load(away_file)
    away_file.close()

    TextBlobSentiment(home_tweets,away_tweets,index)
    VaderSentiment(home_tweets, away_tweets, index)


### SENTIMENT ANALYSIS FUNCTIONS
# VADER SENTIMENT
def VaderSentiment(home_tweets, away_tweets, index):
    # Positive and Negative Tweets Count
    home_pos_vader, home_neg_vader = 0, 0
    away_pos_vader, away_neg_vader = 0, 0
    
    # home
    for k in home_tweets.keys():
        analysis = SentimentIntensityAnalyzer()
        sentiment_dict = analysis.polarity_scores(home_tweets[k])
        # Count Positive/Negative Tweets
        if (sentiment_dict['compound'] >= 0.05):
            home_pos_vader += 1
        elif (sentiment_dict['compound'] <= -0.05):
            home_neg_vader += 1
    
    vader_home_pos = home_pos_vader / len(home_tweets.keys())
    vader_home_neg = home_neg_vader / len(home_tweets.keys())
    
    sentiment_features[index][4] = vader_home_pos
    sentiment_features[index][5] = vader_home_neg

    # away
    for k in away_tweets.keys():
        analysis = SentimentIntensityAnalyzer()
        sentiment_dict = analysis.polarity_scores(away_tweets[k])
        # Count Positive/Negative Tweets
        if (sentiment_dict['compound'] >= 0.05):
            away_pos_vader += 1
        elif (sentiment_dict['compound'] <= -0.05):
            away_neg_vader += 1
    
    vader_away_pos = away_pos_vader / len(away_tweets.keys())
    vader_away_neg = away_neg_vader / len(away_tweets.keys())

    sentiment_features[index][6] = vader_away_pos
    sentiment_features[index][7] = vader_away_neg


# TEXTBLOB SENTIMENT
def TextBlobSentiment(home_tweets,away_tweets,index):
    # Positive and Negative Tweets Count
    home_pos_tweets, home_neg_tweets = 0, 0
    away_pos_tweets, away_neg_tweets = 0, 0

    # Sentiment with TextBlob
    for k in home_tweets.keys():
        analysis = TextBlob(home_tweets[k])
        analysis = analysis[0:-1]
        polar = analysis.sentiment.polarity
        # Count Positive/Negative Tweets
        if (polar > 0.05):
            home_pos_tweets += 1
        elif (polar < -0.05):
            home_neg_tweets += 1
    
    blob_home_pos = home_pos_tweets / len(home_tweets.keys())
    blob_home_neg = home_neg_tweets / len(home_tweets.keys())

    sentiment_features[index][0] = blob_home_pos
    sentiment_features[index][1] = blob_home_neg

    for k in away_tweets.keys():
        analysis = TextBlob(away_tweets[k])
        analysis = analysis[0:-1]
        polar = analysis.sentiment.polarity
        # Count Positive/Negative Tweets
        if (polar > 0.05):
            away_pos_tweets += 1
        elif (polar < -0.05):
            away_neg_tweets += 1
    
    blob_away_pos = away_pos_tweets / len(away_tweets.keys())
    blob_away_neg = away_neg_tweets / len(away_tweets.keys())
    
    sentiment_features[index][2] = blob_away_pos
    sentiment_features[index][3] = blob_away_neg 

### TRAINING FIT
X_train = sentiment_features
Y_train = training_labels

nb = GaussianNB().fit(X_train, Y_train)
svm = LinearSVC(max_iter=12000).fit(X_train, Y_train)
log_reg = LogisticRegression().fit(X_train, Y_train)


#####                                                                               TESTING SECTION


### FORMAT OF SENTIMENT_FEATURES:
# [blob_home_pos, blob_home_neg, blob_away_pos, blob_away_neg, vader_home_pos, vader_home_neg, vader_away_pos, vader_away_neg]
test_sentiment_features = [[0,0,0,0,0,0,0,0] for i in range(len(testing_games))]

### READ IN DATA FROM JSON
def TestReadInFile(index):
    home_path = "dataset/home_test/home_test"+str(index)+".json"
    home_file = open(home_path)
    home_tweets = json.load(home_file)
    home_file.close()

    away_path = "dataset/away_test/away_test"+str(index)+".json"
    away_file = open(away_path)
    away_tweets = json.load(away_file)
    away_file.close()

    TestTextBlobSentiment(home_tweets,away_tweets,index)
    TestVaderSentiment(home_tweets, away_tweets, index)


### SENTIMENT ANALYSIS FUNCTIONS
# VADER SENTIMENT
def TestVaderSentiment(home_tweets, away_tweets, index):
    # Positive and Negative Tweets Count
    home_pos_vader, home_neg_vader = 0, 0
    away_pos_vader, away_neg_vader = 0, 0
    
    # home
    for k in home_tweets.keys():
        analysis = SentimentIntensityAnalyzer()
        sentiment_dict = analysis.polarity_scores(home_tweets[k])
        # Count Positive/Negative Tweets
        if (sentiment_dict['compound'] >= 0.05):
            home_pos_vader += 1
        elif (sentiment_dict['compound'] <= -0.05):
            home_neg_vader += 1
    
    vader_home_pos = home_pos_vader / len(home_tweets.keys())
    vader_home_neg = home_neg_vader / len(home_tweets.keys())
    
    test_sentiment_features[index][4] = vader_home_pos
    test_sentiment_features[index][5] = vader_home_neg

    # away
    for k in away_tweets.keys():
        analysis = SentimentIntensityAnalyzer()
        sentiment_dict = analysis.polarity_scores(away_tweets[k])
        # Count Positive/Negative Tweets
        if (sentiment_dict['compound'] >= 0.05):
            away_pos_vader += 1
        elif (sentiment_dict['compound'] <= -0.05):
            away_neg_vader += 1
    
    vader_away_pos = away_pos_vader / len(away_tweets.keys())
    vader_away_neg = away_neg_vader / len(away_tweets.keys())

    test_sentiment_features[index][6] = vader_away_pos
    test_sentiment_features[index][7] = vader_away_neg


# TEXTBLOB SENTIMENT
def TestTextBlobSentiment(home_tweets,away_tweets,index):
    # Positive and Negative Tweets Count
    home_pos_tweets, home_neg_tweets = 0, 0
    away_pos_tweets, away_neg_tweets = 0, 0

    # Sentiment with TextBlob
    for k in home_tweets.keys():
        analysis = TextBlob(home_tweets[k])
        analysis = analysis[0:-1]
        polar = analysis.sentiment.polarity
        # Count Positive/Negative Tweets
        if (polar > 0.05):
            home_pos_tweets += 1
        elif (polar < -0.05):
            home_neg_tweets += 1
    
    blob_home_pos = home_pos_tweets / len(home_tweets.keys())
    blob_home_neg = home_neg_tweets / len(home_tweets.keys())

    test_sentiment_features[index][0] = blob_home_pos
    test_sentiment_features[index][1] = blob_home_neg

    for k in away_tweets.keys():
        analysis = TextBlob(away_tweets[k])
        analysis = analysis[0:-1]
        polar = analysis.sentiment.polarity
        # Count Positive/Negative Tweets
        if (polar > 0.05):
            away_pos_tweets += 1
        elif (polar < -0.05):
            away_neg_tweets += 1
    
    blob_away_pos = away_pos_tweets / len(away_tweets.keys())
    blob_away_neg = away_neg_tweets / len(away_tweets.keys())
    
    test_sentiment_features[index][2] = blob_away_pos
    test_sentiment_features[index][3] = blob_away_neg 

### CALLING MAIN FUNCTION
def iteration():
    for i in range(len(training_games)):
        ReadInFile(i)
    for i in range(len(testing_games)):
        TestReadInFile(i)
iteration()

### PRINTING RESULT OF SENTIMENT FEATURES BY ITSELF
print("Naive-Bayes")
predicted = nb.predict(test_sentiment_features)
print(metrics.classification_report(testing_labels, predicted))

print("SVM")
predicted = svm.predict(test_sentiment_features)
print(metrics.classification_report(testing_labels, predicted))

print("Logistic regression")
predicted = log_reg.predict(test_sentiment_features)
print(metrics.classification_report(testing_labels, predicted))
