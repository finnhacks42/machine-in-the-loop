import argparse
import sqlite3
import json
import numpy as np
from tqdm import tqdm
import csv
import random

def get_negatives(all_tweets, theme):
    negs = []
    for _theme in all_tweets:
        if theme != _theme:
            negs += all_tweets[_theme]
    return negs

def main(args):

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    query = 'SELECT id, name FROM theme'
    cur.execute(query)
    themes = cur.fetchall()

    all_tweets = {}
    for (_id, _theme) in themes:
        if _theme.startswith('KMeans') or _theme == 'Unknown':
            continue
        all_tweets[_theme] = []

        table_name = 'tweet'
        if args.is_immi:
            table_name = 'tweet_immi'

        query = 'SELECT text FROM {} WHERE theme_id = {} AND good = False ORDER BY distance'
        query = query.format(table_name, _id)
        cur.execute(query)

        tweets = [t[0] for t in cur.fetchall()]
        n = len(tweets)
        n_split = int(n/4)

        #print(_theme, n)
        for i in range(0, n-3, n_split):
            #print(i, len(tweets[i:i+3]))
            all_tweets[_theme] += tweets[i:i+3]

    #exit()
    # Create examples. One negative example for each positive example
    examples = []
    for theme in all_tweets:
        print(theme, len(all_tweets[theme]))
        negatives = get_negatives(all_tweets, theme)
        for i in range(0, len(all_tweets[theme])):
            random_index = random.randint(0,len(negatives)-1)
            neg = negatives[random_index]

            examples.append([all_tweets[theme][i], theme, 1])

            if args.generate_neg:
                examples.append([neg, theme, 0])

    print(len(examples))

    # Create dataset
    random.shuffle(examples)
    with open(args.output_file, 'w') as fp:
        spamwriter = csv.writer(fp, delimiter=';')
        for row in examples:
            spamwriter.writerow(row)

if __name__ == "__main__":
    random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--generate_neg', default=False, action='store_true')
    parser.add_argument('--is_immi', default=False, action='store_true')
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args)
