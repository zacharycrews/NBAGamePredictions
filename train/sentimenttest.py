import snscrape.modules.twitter as sntwitter
import pandas as pd
import json

# Creating list to append tweet data to
tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('Golden State Warriors since:2022-10-18 until:2022-10-19').get_items()):
    if i>500:
        break
    tweets_list2.append([tweet.date, tweet.content, tweet.user.username, tweet.user.verified])

# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet', 'Username', 'Verified'])
print(tweets_df2.head())
