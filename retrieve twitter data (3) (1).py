import tweepy
import csv
import time

consumer_key = 'BlHVe4NazxtdbPgx386iUV6MZ'
consumer_secret = 'f2Hhq39zzPeIaU3mQZgqki2he4ncTXEE8dOglTJtDOl193LK3l'
access_token = '2939760481-Or1d4iggFvuiDkdFjA9Pb3LPURMs5KQMZStzouP'
token_secret = 'FyxTrnXC8bIdImFvWgBesTdeCMfGJ737ZxWM4Igj2TpR0'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

keyword = str(input("Enter a search query "))

csvFile = open('data.csv', 'a')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['Sno.', 'user id', 'user name', 'tweet id', 'text', 'retweet', 'time'])

j = 1
maxid = -1
while True:
    try:
        if maxid == -1:
            lis = tweepy.Cursor(api.search, q=keyword, lang='en').items()
        else:
            lis = tweepy.Cursor(api.search, q=keyword, lang='en', max_id=maxid).items()

        for i in lis:
            if i.retweet_count > 0:
                print([j, i.user.id, i.user.screen_name.encode('utf-8'), i.id, i.text.encode('utf-8', 'ignore'), i.retweet_count, i.created_at])
                csvWriter.writerow([j, i.user.id, i.user.screen_name.encode('utf-8'), i.id, i.text.encode('utf-8', 'ignore'), i.retweet_count, i.created_at])
                maxid = int(i.id) - 1
                j += 1
            if j >= 100000:
                break

        if j >= 100000:
            break
    except:
        print(time.localtime(time.time()))
        time.sleep(300)
csvFile.close()
