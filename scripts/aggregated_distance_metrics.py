import argparse
import sqlite3
import json
import numpy as np
from tqdm import tqdm
import csv
import random
import pandas as pd


def get_theme_quartiles(cur, theme_id, table_name):
    Q1 = []; Q2 = []; Q3 = []; Q4 = []
    query = 'SELECT tweet_id,distance FROM {} WHERE theme_id = {} AND good = False ORDER BY distance'
    query = query.format(table_name, theme_id)
    cur.execute(query)

    tweets = [t for t in cur.fetchall()]
    distances = [t[1] for t in tweets]
    q1_num = np.percentile(distances, 25)
    q2_num = np.percentile(distances, 50)
    q3_num = np.percentile(distances, 75)
    #print(q1_num, q2_num, q3_num)

    n = len(tweets)

    Q1 += [t for t in tweets if t[1] <= q1_num]
    #print(len(tweets[:offset+n_split]))

    Q2 += [t for t in tweets if t[1] <= q2_num]
    #print(len(tweets[:offset+n_split]))

    Q3 += [t for t in tweets if t[1] <= q3_num]
    #print(len(tweets[:offset+n_split]))

    Q4 += tweets

    return (Q1, Q2, Q3, Q4)

def get_all_quartiles(cur, isimmi=False):
    query = 'SELECT id, name FROM theme'
    cur.execute(query)
    themes = cur.fetchall()

    # Name of table
    table_name = 'tweet'
    if isimmi:
        table_name = 'tweet_immi'

    Q1 =  []; Q2 = []; Q3 = []; Q4 = []
    for (_id, _theme) in themes:
        if _theme.startswith('KMeans') or _theme == 'Unknown':
            continue

        (_Q1, _Q2, _Q3, _Q4) = get_theme_quartiles(cur, _id, table_name)
        print(_theme, len(_Q1))
        Q1 += _Q1
        Q2 += _Q2
        Q3 += _Q3
        Q4 += _Q4
    return (Q1, Q2, Q3, Q4)

def main(args):

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    (Q1, Q2, Q3, Q4) = get_all_quartiles(cur)

    # Keep distances only
    Q1 = [t[1] for t in Q1]
    Q2 = [t[1] for t in Q2]
    Q3 = [t[1] for t in Q3]
    Q4 = [t[1] for t in Q4]

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
