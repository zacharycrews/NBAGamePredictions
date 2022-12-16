from ReadCSV import training_games, training_labels, testing_games, testing_labels
from textblob import TextBlob
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import json

month_lengths = {10:31,11:30,12:31,1:31,2:28,3:31,4:30,5:31,6:30}


### TRAINING FEATURES ###

game_features = [[0,0,0,0,0,0,0,0,0,0] for i in range(len(training_games))]

for i, game in enumerate(training_games):
    home_team = game['home']
    away_team = game['visitor']

    # Calculate ppg and papg
    home_pts = 0
    home_pa = 0
    home_games = 0
    away_pts = 0
    away_pa = 0
    away_games = 0
    for stats in training_games[:i]:
        if stats['home'] == home_team:
            home_pts += int(stats['home_PTS'])
            home_pa += int(stats['visitor_PTS'])
            home_games += 1
        elif stats['visitor'] == home_team:
            home_pts += int(stats['visitor_PTS'])
            home_pa += int(stats['home_PTS'])
            home_games += 1
        if stats['home'] == away_team:
            away_pts += int(stats['home_PTS'])
            away_pa += int(stats['visitor_PTS'])
            away_games += 1
        elif stats['visitor'] == away_team:
            away_pts += int(stats['visitor_PTS'])
            away_pa += int(stats['home_PTS'])
            away_games += 1

    game_features[i][2] = home_pts / max(home_games,1)
    game_features[i][3] = home_pa / max(home_games,1)
    game_features[i][7] = away_pts / max(away_games,1)
    game_features[i][8] = away_pa / max(away_games,1)

    # Determine if it is team's second game in a row
    start_date = game['start_date_time'].split("-")
    current_date = start_date.copy()
    current_date = "-".join(current_date[:3])
    if int(start_date[2]) > 0:
        start_date[2] = str(int(start_date[2]) - 1)
        previous_date = "-".join(start_date[:3])
    else:
        if int(start_date[1]) == 1:         # New Years
            start_date[0] = str(int(start_date[0]) - 1)
            start_date[1] = "12"
            start_date[2] = "30"
            previous_date = "-".join(start_date[:3])
        else:                               # First day of month
            start_date[1] = str(int(start_date[1]) - 1)
            start_date[2] = str(month_lengths[int(start_date[1])] - 1)
            previous_date = "-".join(start_date[:3])
    for j in range(i-1, -1, -1):
        prev_game = training_games[j]
        if previous_date in prev_game['start_date_time'] or current_date in prev_game['start_date_time']:
            if (prev_game['home'] == home_team or prev_game['visitor'] == home_team) and previous_date in prev_game['start_date_time']:
                game_features[i][4] = 1
            if (prev_game['home'] == away_team or prev_game['visitor'] == away_team) and previous_date in prev_game['start_date_time']:
                game_features[i][9] = 1
        else:
            break
        
    # Get sentiment score
    # Positive and Negative Tweets Count
    home_pos_tweets, home_neg_tweets = 0, 0
    away_pos_tweets, away_neg_tweets = 0, 0

    # Read in json dataset
    home_path = "dataset/home_train/home_train"+str(i)+".json"
    home_file = open(home_path)
    home_tweets = json.load(home_file)
    home_file.close()

    away_path = "dataset/away_train/away_train"+str(i)+".json"
    away_file = open(away_path)
    away_tweets = json.load(away_file)
    away_file.close()

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

    home_percent_pos = home_pos_tweets / len(home_tweets.keys())
    home_percent_neg = home_neg_tweets / len(home_tweets.keys())

    game_features[i][0] = home_percent_pos
    game_features[i][1] = home_percent_neg

    for k in away_tweets.keys():
        analysis = TextBlob(away_tweets[k])
        analysis = analysis[0:-1]
        polar = analysis.sentiment.polarity

        # Count Positive/Negative Tweets
        if (polar > 0.05):
            away_pos_tweets += 1
        elif (polar < -0.05):
            away_neg_tweets += 1

    away_percent_pos = away_pos_tweets / len(away_tweets.keys())
    away_percent_neg = away_neg_tweets / len(away_tweets.keys())

    game_features[i][5] = away_percent_pos
    game_features[i][6] = away_percent_neg 


X_train = game_features
Y_train = training_labels

nb = GaussianNB().fit(X_train, Y_train)
svm = LinearSVC(max_iter=12000).fit(X_train, Y_train)
log_reg = LogisticRegression().fit(X_train, Y_train)
neural = MLPClassifier(batch_size=20,random_state=1,learning_rate='adaptive').fit(X_train, Y_train)

### TESTING FEATURES ###

test_features = [[0,0,0,0,0,0,0,0,0,0] for i in range(len(testing_games))]

for i, game in enumerate(testing_games):
    home_team = game['home']
    away_team = game['visitor']

    # Calculate ppg and papg
    home_pts = 0
    home_pa = 0
    home_games = 0
    away_pts = 0
    away_pa = 0
    away_games = 0
    for stats in testing_games[:i]:
        if stats['home'] == home_team:
            home_pts += int(stats['home_PTS'])
            home_pa += int(stats['visitor_PTS'])
            home_games += 1
        elif stats['visitor'] == home_team:
            home_pts += int(stats['visitor_PTS'])
            home_pa += int(stats['home_PTS'])
            home_games += 1
        if stats['home'] == away_team:
            away_pts += int(stats['home_PTS'])
            away_pa += int(stats['visitor_PTS'])
            away_games += 1
        elif stats['visitor'] == away_team:
            away_pts += int(stats['visitor_PTS'])
            away_pa += int(stats['home_PTS'])
            away_games += 1

    test_features[i][2] = home_pts / max(home_games,1)
    test_features[i][3] = home_pa / max(home_games,1)
    test_features[i][7] = away_pts / max(away_games,1)
    test_features[i][8] = away_pa / max(away_games,1)

    # Determine if it is team's second game in a row
    start_date = game['start_date_time'].split("-")
    current_date = start_date.copy()
    current_date = "-".join(current_date[:3])
    if int(start_date[2]) > 0:
        start_date[2] = str(int(start_date[2]) - 1)
        previous_date = "-".join(start_date[:3])
    else:
        if int(start_date[1]) == 1:         # New Years
            start_date[0] = str(int(start_date[0]) - 1)
            start_date[1] = "12"
            start_date[2] = "30"
            previous_date = "-".join(start_date[:3])
        else:                               # First day of month
            start_date[1] = str(int(start_date[1]) - 1)
            start_date[2] = str(month_lengths[int(start_date[1])] - 1)
            previous_date = "-".join(start_date[:3])
    for j in range(i-1, -1, -1):
        prev_game = testing_games[j]
        if previous_date in prev_game['start_date_time'] or current_date in prev_game['start_date_time']:
            if (prev_game['home'] == home_team or prev_game['visitor'] == home_team) and previous_date in prev_game['start_date_time']:
                game_features[i][4] = 1
            if (prev_game['home'] == away_team or prev_game['visitor'] == away_team) and previous_date in prev_game['start_date_time']:
                game_features[i][9] = 1
        else:
            break
        
    # Get sentiment score
    # Positive and Negative Tweets Count
    home_pos_tweets, home_neg_tweets = 0, 0
    away_pos_tweets, away_neg_tweets = 0, 0

    # Read in json dataset
    home_path = "dataset/home_test/home_test"+str(i)+".json"
    home_file = open(home_path)
    home_tweets = json.load(home_file)
    home_file.close()

    away_path = "dataset/away_test/away_test"+str(i)+".json"
    away_file = open(away_path)
    away_tweets = json.load(away_file)
    away_file.close()

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

    home_percent_pos = home_pos_tweets / len(home_tweets.keys())
    home_percent_neg = home_neg_tweets / len(home_tweets.keys())

    test_features[i][0] = home_percent_pos
    test_features[i][1] = home_percent_neg

    for k in away_tweets.keys():
        analysis = TextBlob(away_tweets[k])
        analysis = analysis[0:-1]
        polar = analysis.sentiment.polarity

        # Count Positive/Negative Tweets
        if (polar > 0.05):
            away_pos_tweets += 1
        elif (polar < -0.05):
            away_neg_tweets += 1

    away_percent_pos = away_pos_tweets / len(away_tweets.keys())
    away_percent_neg = away_neg_tweets / len(away_tweets.keys())

    test_features[i][5] = away_percent_pos
    test_features[i][6] = away_percent_neg 


print("Naive-Bayes")
nb_predicted = nb.predict(test_features)
print(metrics.classification_report(testing_labels, nb_predicted))

print("SVM")
svm_predicted = svm.predict(test_features)
print(metrics.classification_report(testing_labels, svm_predicted))

print("Logistic regression")
log_reg_predicted = log_reg.predict(test_features)
print(metrics.classification_report(testing_labels, log_reg_predicted))

print("Feed-forward NN")
neural_predicted = neural.predict(test_features)
print(metrics.classification_report(testing_labels, neural_predicted))

