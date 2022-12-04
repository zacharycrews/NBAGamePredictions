from ReadCSV import training_games
from ReadCSV import training_labels
import textblob
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

month_lengths = {10:31,11:30,12:31,1:31,2:28,3:31,4:30,5:31,6:31}

### TRAIN MODEL ###

game_features = [[0,0,0,0,0,0,0,0] for i in range(len(training_games))]

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
                game_features[i][3] = 1
            if (prev_game['home'] == away_team or prev_game['visitor'] == away_team) and previous_date in prev_game['start_date_time']:
                game_features[i][7] = 1
        else:
            break
        
    # Get sentiment score


for i in range(100):
    print(str(i) + ":\t" + str(game_features[i]))
X_train = game_features
Y_train = training_labels

svm = SVC().fit(X_train, Y_train)
nb = GaussianNB().fit(X_train, Y_train)
log_reg = LogisticRegression().fit(X_train, Y_train)






