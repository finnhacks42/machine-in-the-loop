import argparse
import sqlite3
import json
import numpy as np
from tqdm import tqdm
import csv
import random
import pandas as pd


def main(args):

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    query = 'SELECT id, name FROM theme'
    cur.execute(query)
    themes = cur.fetchall()

    all_tweets = {}
    Q1 =  []; Q2 = []; Q3 = []; Q4 = []
    for (_id, _theme) in themes:
        if _theme.startswith('KMeans') or _theme == 'Unknown':
            continue
        all_tweets[_theme] = []

        query = 'SELECT text,distance FROM tweet_immi WHERE theme_id = {} AND good = False ORDER BY distance'
        query = query.format(_id)
        cur.execute(query)

        tweets = [t[1] for t in cur.fetchall()]
        n = len(tweets)
        n_split = int(n/4)
        print(_theme, n, n_split)

        offset = 0
        Q1 += tweets[:offset+n_split]
        print(len(tweets[:offset+n_split]))

        offset += n_split
        Q2 += tweets[:offset+n_split]
        print(len(tweets[:offset+n_split]))

        offset += n_split
        Q3 += tweets[:offset+n_split]
        print(len(tweets[:offset+n_split]))


        offset += n_split
        Q4 += tweets[:n]
        print(len(tweets[:n]))

    print("###############")

    print("Q1")
    df_describe = pd.DataFrame(Q1)
    print(df_describe.describe())
    print('-------------------')

    print("Q2")
    df_describe = pd.DataFrame(Q2)
    print(df_describe.describe())
    print('-------------------')

    print("Q3")
    df_describe = pd.DataFrame(Q3)
    print(df_describe.describe())
    print('-------------------')

    print("Q4")
    df_describe = pd.DataFrame(Q4)
    print(df_describe.describe())
    print('###################')


    #ALL =  Q1 + Q2 + Q3 +Q4
    #df_describe=pd.DataFrame(ALL)
    #print(df_describe.describe())

    print("{}, {}, {}, {}".format(len(Q1), len(Q2), len(Q3), len(Q4)))


if __name__ == "__main__":
    random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    args = parser.parse_args()
    main(args)
