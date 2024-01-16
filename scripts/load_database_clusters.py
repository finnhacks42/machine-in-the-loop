import argparse
import sqlite3
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def parse_predicate(pred):
    name, args = pred.split('(')
    args = args[:-1].split(',')
    return name, args

def calculate_score(centroids, embeds, pos_id):
    centroid_distances = []
    for centroid in centroids:
        cos_score = util.cos_sim(centroid, embeds[pos_id])[0][0]
        centroid_distances.append(cos_score)
    max_cos_score = max(centroid_distances)
    return 1 - max_cos_score.item()

def main(args):
    embedder = SentenceTransformer('all-mpnet-base-v2')
    curr_embeds = np.load(args.sbert)

    with open(args.id2pos) as fp:
        id2pos = json.load(fp)

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    # embed all phrases
    query = 'SELECT id, name from theme'
    cur.execute(query)
    themes = [(a,b) for (a,b) in cur.fetchall() if not b.startswith('KMeans') and b != 'Unknown']
    theme_centroids = {}
    for (theme_id, theme_name) in themes:
        cur.execute('select text from phrase where goodness = "good" and theme_id = {}'.format(theme_id))
        phrases = cur.fetchall()
        phrases = [ph[0] for ph in phrases]
        centroids = embedder.encode(phrases)
        theme_centroids[theme_id] = centroids

    # read predictions and update data
    mentions_argument = set()
    with open(args.input_preds) as fp:
        lines = fp.readlines()
        pbar = tqdm(total=len(lines), desc='reading predictions and update clusters...')

        for line in lines:
            name, args = parse_predicate(line.strip())
            if name == "MentionsArgument":
                twid = args[0]
                theme = args[1]

                if theme.startswith('KMeans') or theme == 'Unknown':
                    continue

                mentions_argument.add(twid)
                positions = id2pos[twid]

                query = 'SELECT id from theme where name = "{}"'
                query = query.format(theme)
                cur.execute(query)
                theme_id = cur.fetchone()[0]

                for pos in positions:
                    distance = calculate_score(theme_centroids[theme_id], curr_embeds, pos)
                    query = 'UPDATE tweet SET theme_id = {}, distance = {} where tweet_id = {}'
                    query = query.format(theme_id, distance, pos)
                    cur.execute(query)

            pbar.update(1)
        pbar.close()

    pbar = tqdm(total=len(id2pos), desc='marking unknowns...')

    query = 'SELECT id from theme where name = "Unknown"'
    query = query.format(theme)
    cur.execute(query)
    unknown = cur.fetchone()

    if unknown is None:
        # create it and re-select it
        query = 'INSERT INTO theme (name) VALUES ("Unknown")'
        cur.execute(query)
        unknown = cur.fetchone()

        query = 'SELECT id from theme where name = "Unknown"'
        query = query.format(theme)
        cur.execute(query)
        unknown = cur.fetchone()

    theme_id = unknown[0]

    for twid in id2pos:
        if twid not in mentions_argument:
            positions = id2pos[twid]

            for pos in positions:
                query = 'UPDATE tweet SET theme_id = {} where tweet_id = {}'
                query = query.format(theme_id, pos)
                cur.execute(query)
        pbar.update(1)
    pbar.close()

    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_preds', type=str, required=True)
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--id2pos', type=str, required=True)
    parser.add_argument('--sbert', type=str, required=True)
    args = parser.parse_args()
    main(args)
