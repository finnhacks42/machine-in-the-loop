tweets_is_tweet = set(); tweets_is_tweet_list = []
tweets_immi_frame = set(); tweets_immi_frame_list = []
tweets_political_frame = set()


with open('drail_programs/data_immi/is_tweet.txt') as fp:
    for line in fp:
        tw = line.strip()
        tweets_is_tweet_list.append(tw)
        tweets_is_tweet.add(tw)

print(len(tweets_is_tweet_list))
print(len(tweets_is_tweet))

with open('drail_programs/data_immi/has_immi_frame.txt') as fp:
    for line in fp:
        tw, frame = line.strip().split('\t')
        #print(tw, frame)
        tweets_immi_frame.add(tw)
        tweets_immi_frame_list.append(tw)

print(len(tweets_immi_frame_list))
print(len(tweets_immi_frame))

