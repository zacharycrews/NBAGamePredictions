


months = {"Jan" : '1', "Feb" : '2', "Mar" : '3', "Apr" : '4', "May" : '5',
          "Jun" : '6', "Jul" : '7', "Aug" : '8', "Sep" : '9', "Oct" : '10',
          "Nov" : '11', "Dec" : '12'}


# 1 = home team won, 0 = away team won
training_labels = []
testing_labels = []

# This list contains a dictionary with start/end times, teams, etc.
training_games = []
testing_games = []


f = open("train/training.csv", "r")
next(f)
for x in f:
    dic = {}
    llist = x.split(",")


    #Getting start and end dates and adjusting formats
    llist[0] = llist[0].split(" ")
    llist[0].pop(0)
    llist[0][0] = months[llist[0][0]]
    llist[0][0], llist[0][1], llist[0][2] = llist[0][2], llist[0][0], llist[0][1]


    start_date = [llist[0][0], llist[0][1], str(int(llist[0][2]) - 1)]
    start_date = '-'.join(start_date)

    end_date = '-'.join(llist[0])

    
    # Getting game time and converting to 24 hour time
    time = llist[1].split(':')
    if time[-1][-1] == 'p':
        time[-1] = time[-1][0 : -1]
        time[0] = str(int(time[0]) + 12)
    else:
        time[-1] = time[-1][0 : -1]
        if len(time[0]) < 2:
            time[0] = '0' + time[0]

    time = '-'.join(time)


    # Combine start/end dates with time to pass into twitter scraper
    start_date_time = start_date + '-' +  time
    end_date_time = end_date + '-' + time
    dic['start_date_time'] = start_date_time
    dic['end_date_time'] = end_date_time



    # Vistor team and points scored
    dic['visitor'] = llist[2]
    dic['visitor_PTS'] = llist[3]

    # Home team and points scored
    dic['home'] = llist[4]
    dic['home_PTS'] = llist[5]


    training_games.append(dic)

    # Who won?
    if int(dic['visitor_PTS']) >= int(dic['home_PTS']):
        training_labels.append(0)
    else:
        training_labels.append(1)    
f.close()

f = open("test/game_results.csv", "r")
next(f)
for x in f:
    dic = {}
    llist = x.split(",")


    #Getting start and end dates and adjusting formats
    llist[0] = llist[0].split(" ")
    llist[0].pop(0)
    llist[0][0] = months[llist[0][0]]
    llist[0][0], llist[0][1], llist[0][2] = llist[0][2], llist[0][0], llist[0][1]


    start_date = [llist[0][0], llist[0][1], str(int(llist[0][2]) - 1)]
    start_date = '-'.join(start_date)

    end_date = '-'.join(llist[0])

    
    # Getting game time and converting to 24 hour time
    time = llist[1].split(':')
    if time[-1][-1] == 'p':
        time[-1] = time[-1][0 : -1]
        time[0] = str(int(time[0]) + 12)
    else:
        time[-1] = time[-1][0 : -1]
        if len(time[0]) < 2:
            time[0] = '0' + time[0]

    time = '-'.join(time)


    # Combine start/end dates with time to pass into twitter scraper
    start_date_time = start_date + '-' +  time
    end_date_time = end_date + '-' + time
    dic['start_date_time'] = start_date_time
    dic['end_date_time'] = end_date_time



    # Vistor team and points scored
    dic['visitor'] = llist[2]
    dic['visitor_PTS'] = llist[3]

    # Home team and points scored
    dic['home'] = llist[4]
    dic['home_PTS'] = llist[5]


    testing_games.append(dic)

    # Who won?
    if int(dic['visitor_PTS']) >= int(dic['home_PTS']):
        testing_labels.append(0)
    else:
        testing_labels.append(1)    
f.close()


#print(training_games[0])
#print(training_labels[0])
#print('\n\n')
#print(testing_games[0])
#print(testing_labels[0])




