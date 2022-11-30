import snscrape.modules.twitter as sntwitter
import pandas as pd
import json
import datetime
import calendar

# game_start_dates = []
# game_end_dates = []

games = []

def getGameDates():



    months = {"Jan" : '1', "Feb" : '2', "Mar" : '3', "Apr" : '4', "May" : '5',
              "Jun" : '6', "Jul" : '7', "Aug" : '8', "Sep" : '9', "Oct" : '10',
              "Nov" : '11', "Dec" : '12'}


    # 1 = home team won, 0 = away team won
    labels = []

    # This list contains a dictionary with start/end times, teams, etc


    f = open("training.csv", "r")
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



        date_switch = {
            "2021-11-1-": "2021-10-31-",
            "2021-12-1-": "2021-11-30-",
            "2022-1-1-": "2021-12-31-",
            "2022-2-1-": "2022-1-31-",
            "2022-3-1-": "2022-2-28-",
            "2022-4-1-": "2022-3-31-",
            "2022-5-1-": "2022-4-30-",
            "2022-6-1-": "2022-5-31-"
        }

        for j in date_switch:
            if j in end_date_time:
                start_date_time = end_date_time.replace(j, date_switch[j])
                dic['start_date_time'] = start_date_time
        # if date_switch in end_date_time:


        dic['start_date_time'] = start_date_time
        dic['end_date_time'] = end_date_time

        # Vistor team and points scored
        dic['visitor'] = llist[2]
        dic['visitor_PTS'] = llist[3]

        # Home team and points scored
        dic['home'] = llist[4]
        dic['home_PTS'] = llist[5]


        games.append(dic)

        # Who won?
        if dic['visitor_PTS'] >= dic['home_PTS']:
            labels.append(0)
        else:
            labels.append(1)
    f.close()


    #Example of game list and label list
    # for j in range(len(games)):
    #     game_start_dates.append(games[j]["start_date_time"])
    #     game_end_dates.append(games[j]["end_date_time"])
    # print(labels[0])

def getTweets(games):
    print(games)
    for q in range(10): #len(game_start_dates)

        # Creating list to append tweet data to
        tweets_list_home = []
        tweets_list_away = []

        #process each game data
        syear, smonth, sday, shour, smin = games[q]["start_date_time"].split('-')
        syear = int(syear)
        smonth = int(smonth)
        sday = int(sday)
        shour = int(shour)
        smin = int(smin)
        eyear, emonth, eday, ehour, emin = games[q]["end_date_time"].split('-')
        eyear = int(eyear)
        emonth = int(emonth)
        eday = int(eday)
        ehour = int(ehour)
        emin = int(emin)
        home_team = games[q]["home"]
        away_team = games[q]["visitor"]

        # Get Starting time of game (year,month,date,hour,min) -- important: (24 hour time)
        t_begin_search = int(datetime.datetime(syear,smonth,sday,shour,smin).timestamp())
        # Get 24 hours BEFORE the start time of game (year,month,date,hour,min) -- important: (24 hour time)
        t_end_search = int(datetime.datetime(eyear,emonth,eday,ehour,emin).timestamp())

        # EDGE CASES : if a game is played on the first of the month, we need to make sure that we get the
        # previous date correct. As in, a game played on December 1st needs to begin searching for tweets
        # starting at November 30th

        # should be able to iterate over the entire dataframe when read in, so a nested foreloop outside the one
        # below, using the integer value in the df, then feeding the datetime in the first column to a function
        # created out of the code above to convert to epoch, then someting like df[index]['Visitor/Neutral'] in one
        # loop to get the tweets for the away team, then the same process for df[index]['Home/Neutral'] for another loop
        # then we can store this information... in a new dataframe? this would be called something like "testweets.csv"?
        # where we include what information in the dataframe... just the tweets, username, verified status? Then
        # Dex and Zach can take the tweets and the informaiton we scraped to use for the model and predict sentiment?
        # But from here, that leaves them only the sentiment as an inclusion in their training data... is there something
        # more we need to give them? Surely home vs. away will be included as a parameter in their data, but what else
        # could we give them to include in their dataset? not sure...

        # Using TwitterSearchScraper to scrape data and append tweets to list
        outfile_home = open('hometweets/home{}.txt'.format(q), 'w')
        outfile_away = open('awaytweets/away{}.txt'.format(q), 'w')
        # for loop for scraping all tweets for the HOME team
        # note, for both cases, the first parameter in format is going to be the name of the team we are scraping, likely to come from the df we import from csv
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('{} since_time:{} until_time:{} lang:en'.format(home_team, t_begin_search, t_end_search)).get_items()):
            # tweets_list_home.append([tweet.date, tweet.content, tweet.user.username, tweet.user.verified])
            # tweets_list_home.append(str(tweet.content))
            outfile_home.write(str(tweet.content).strip())
        print('completed')
        for j,tweet2 in enumerate(sntwitter.TwitterSearchScraper('{} since_time:{} until_time:{} lang:en'.format(away_team, t_begin_search, t_end_search)).get_items()):
            # tweets_list_away.append([tweet2.date, tweet2.content, tweet2.user.username, tweet2.user.verified])
            outfile_away.write(str(tweet2.content).strip())

        # Creating a dataframe from the tweets list above
        # tweets_df_home = pd.DataFrame(tweets_list_home, columns=['Datetime', 'Tweet', 'Username', 'Verified'])
        # tweets_df_away = pd.DataFrame(tweets_list_away, columns=['Datetime', 'Tweet', 'Username', 'Verified'])
        # print(tweets_df_home.tail())
        # print()
        # print(tweets_df_away.tail())
        # print(tweets_list_home)

getGameDates()
print(len(games))
# getTweets(games)
