import argparse
import json
import sqlite3
from tqdm import tqdm

def main(args):
    # Neuro-Sym database (second interaction)
    conn_ns = sqlite3.connect(args.database_ns)
    cur_ns = conn_ns.cursor()

    # Nearest-neighs database
    conn_nn = sqlite3.connect(args.database_nn)
    cur_nn = conn_nn.cursor()

    # Query all themes and good examples from Neuro-Sym DB
    query = 'SELECT id, name from theme'
    cur_ns.execute(query)
    themes = [(a,b) for (a,b) in cur_ns.fetchall() if not b.startswith('KMeans') and b != 'Unknown']

    for (theme_id, theme_name) in themes:

        # Check if theme exists in NN database
        query = 'select id, name from theme where name = "{}"'.format(theme_name)
        cur_nn.execute(query)
        _theme = cur_nn.fetchone()

        # If it doesn't exist
        if _theme is None:
            print(theme_id, theme_name)
            # Add it to NN database
            query = 'insert into theme (id, name) values ({}, "{}")'.format(theme_id, theme_name)
            cur_nn.execute(query)

            # Find all phrases from NS DB
            query = 'select id, text, goodness, narrative, immi_frame, immi_role, frame_political, frame_policy, frame_crime, frame_health, frame_security, frame_economic from phrase_immi where theme_id = {}'.format(theme_id)
            cur_ns.execute(query)
            phrases = cur_ns.fetchall()
            print("..Adding {} phrases".format(len(phrases)))

            # Add all pahrases into NN database
            for (phrase_id, text, goodness, narrative, immi_frame, immi_role,\
                frame_political, frame_policy, frame_crime, frame_health, frame_security,\
                frame_economic) in phrases:
                if '"' in text:
                    text = text.replace('"', "'")
                query = '''insert into phrase_immi (id, text, goodness, narrative, immi_frame, immi_role, frame_political, frame_policy, frame_crime, frame_health, frame_security, frame_economic, theme_id) values ({}, "{}", "{}", "{}", "{}", "{}", {}, {}, {}, {}, {}, {}, {})'''
                query = query.format(phrase_id, text, goodness, narrative, immi_frame,
                                     immi_role, frame_political, frame_policy,
                                     frame_crime, frame_health, frame_security,
                                     frame_economic, theme_id)

                cur_nn.execute(query)

            # Find all tweets from NS DB
            query = 'select id, good from tweet_immi where theme_id = {}'.format(theme_id)
            cur_ns.execute(query)
            tweets = cur_ns.fetchall()
            print("..Adding {} tweets".format(len(tweets)))

            # Update tweet in NN DB to be assigned to new theme
            for (tw_id, good) in tweets:
                query = 'update tweet_immi set good = {}, theme_id = {} where id = {}'.format(good, theme_id, tw_id)
                cur_nn.execute(query)

    conn_ns.close()

    conn_nn.commit()
    conn_nn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_ns', type=str, required=True)
    parser.add_argument('--database_nn', type=str, required=True)
    args = parser.parse_args()
    main(args)


