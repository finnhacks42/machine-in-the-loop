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

    N = 0; m_nar = 0; m_role = 0; m_frame = 0; m_pol_frame = 0

    for theme in themes:
        query = 'select tweet_id, narrative, immi_role, immi_frame, frame_political, frame_policy, frame_crime, frame_health, frame_security, frame_economic from tweet_immi where theme_id = {}'.format(theme)
        query = query.format(theme)
        tweets = cur.execute(query).fetchall()

        #print(theme)
        Q1, Q2, Q3, Q4 = get_theme_quartiles(cur, theme, 'tweet_immi')
        Q1 = set([t[0] for t in Q1])
        Q2 = set([t[0] for t in Q2])
        Q3 = set([t[0] for t in Q3])
        Q4 = set([t[0] for t in Q4])

        narratives = []; immi_roles = []; immi_frames = []; policy_frames = []
        n_tweets = set()
        for (tweet_id, n, r, f, polit_f, polic_f, crime_f, health_f, sec_f, eco_f) in tweets:
            # Filter out what we don't want
            if args.quartile == 1 and tweet_id not in Q1:
                continue
            elif args.quartile == 2 and tweet_id not in Q2:
                continue
            elif args.quartile == 3 and tweet_id not in Q3:
                continue

            narratives.append(n)
            immi_roles.append(r)
            immi_frames.append(f)

            if int(polit_f) > 0:
                policy_frames.append('political')
            if int(polic_f) > 0:
                policy_frames.append('policy')
            if int(crime_f) > 0:
                policy_frames.append('crime')
            if int(health_f) > 0:
                policy_frames.append('health')
            if int(sec_f) > 0:
                policy_frames.append('security')
            if int(eco_f) > 0:
                policy_frames.append('economic')

            n_tweets.add(tweet_id)

        N += len(n_tweets)

        # Narrative
        nar_counter = Counter(narratives)
        nar_label = nar_counter.most_common(1)[0]
        m_nar += nar_label[1]

        # Roles
        role_counter = Counter(immi_roles)
        role_label = role_counter.most_common(1)[0]
        m_role += role_label[1]

        # Immi Frame
        frame_counter = Counter(immi_frames)
        frame_label = frame_counter.most_common(1)[0]
        m_frame += frame_label[1]

        # Policy Frames
        pol_frame_counter = Counter(policy_frames)
        pol_frame_label = pol_frame_counter.most_common(1)[0]
        m_pol_frame += pol_frame_label[1]

    narrative_purity = m_nar / N
    role_purity = m_role / N
    immi_frame_purity = m_frame / N
    pol_frame_purity = m_pol_frame / N

    print("Narrative purity", narrative_purity)
    print("Immi Role purity", role_purity)
    print("Immi Frame purity", immi_frame_purity)
    print("Policy Frame purity", pol_frame_purity)

    print("Avg purity", np.mean([narrative_purity, role_purity, immi_frame_purity, pol_frame_purity]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--quartile', type=int, default=None)
    args = parser.parse_args()
    main(args)
