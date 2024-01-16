import csv
import argparse
import numpy as np
import sqlite3
from collections import Counter
from aggregated_distance_metrics import get_theme_quartiles

def main(args):

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    query = 'select id, name from theme'
    themes = cur.execute(query).fetchall()
    themes = [_id for (_id, _name) in themes if not _name.startswith('KMeans')]

    N = 0; m_stance = 0; m_moral = 0; m_mf = 0

    for theme in themes:
        query = 'select tweet_id, stance, morality, mf from tweet where theme_id = {}'.format(theme)
        query = query.format(theme)
        tweets = cur.execute(query).fetchall()

        #print(theme)
        Q1, Q2, Q3, Q4 = get_theme_quartiles(cur, theme, 'tweet')
        Q1 = set([t[0] for t in Q1])
        Q2 = set([t[0] for t in Q2])
        Q3 = set([t[0] for t in Q3])
        Q4 = set([t[0] for t in Q4])


        stances = []; morals = []; mfs = []
        n_tweets = set()
        for (tweet_id, stance, moral, mf) in tweets:
            # Filter out what we don't want
            if args.quartile == 1 and tweet_id not in Q1:
                continue
            elif args.quartile == 2 and tweet_id not in Q2:
                continue
            elif args.quartile == 3 and tweet_id not in Q3:
                continue

            stances.append(stance)
            morals.append(moral)
            mfs.append(mf)
            n_tweets.add(tweet_id)

        N += len(n_tweets)

        # Stance
        stance_counter = Counter(stances)
        stance_label = stance_counter.most_common(1)[0]
        m_stance += stance_label[1]

        # Morality
        moral_counter = Counter(morals)
        moral_label = moral_counter.most_common(1)[0]
        m_moral += moral_label[1]

        # Moral Foundation
        mf_counter = Counter(mfs)
        mf_label = mf_counter.most_common(1)[0]
        m_mf += mf_label[1]

    stance_purity = m_stance / N
    moral_purity = m_moral / N
    mf_purity = m_mf / N

    print("Stance purity", stance_purity)
    print("Morality purity", moral_purity)
    print("MF purity", mf_purity)

    print("Avg purity", np.mean([stance_purity, mf_purity]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--quartile', type=int, default=None)
    args = parser.parse_args()
    main(args)
