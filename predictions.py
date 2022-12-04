from ReadCSV import training_games
from ReadCSV import training_labels
import textblob
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



game_features = [[0,0,0,0,0,0,0,0,0] for i in range(len(training_games))]

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
    game_features[i][1] = home_pts / max(home_games,1)
    game_features[i][2] = home_pa / max(home_games,1)
    game_features[i][5] = away_pts / max(away_games,1)
    game_features[i][6] = away_pa / max(away_games,1)

    # Get sentiment score
    
X_train = game_features
Y_train = training_labels

svm = SVC().fit(X_train, Y_train)
nb = GaussianNB().fit(X_train, Y_train)
log_reg = LogisticRegression().fit(X_train, Y_train)






