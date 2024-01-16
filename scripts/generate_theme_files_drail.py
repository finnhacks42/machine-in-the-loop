import argparse
import sqlite3
import json
import numpy as np
from tqdm import tqdm
import os
import torch
from sentence_transformers import SentenceTransformer, util

def _top_K(phrases, mfs, stances, embedder, embeds, K):
    top_results = []; top_stances = []; top_mfs = []
    if len(phrases) > 0:
        centroids = embedder.encode(phrases)

        centroid_distances = []; centroid_stances = []; centroid_mfs = []
        for centroid, stance, mf in zip(centroids, stances, mfs):
            cos_scores = util.cos_sim(centroid, embeds)[0]
            centroid_distances.append(cos_scores)
            centroid_stances.append(stance)
            centroid_mfs.append(mf)

        out = torch.column_stack(centroid_distances)
        out_max = torch.max(out, 1)

        max_cos_scores = out_max.values
        max_cos_indices = out_max.indices

        res_stances = [centroid_stances[idx] for idx in max_cos_indices]
        res_mfs = [centroid_mfs[idx] for idx in max_cos_indices]

        top_results = torch.topk(max_cos_scores, k=K).indices
        top_stances = [res_stances[idx] for idx in top_results]
        top_mfs = [res_mfs[idx] for idx in top_results]

    return top_results, top_stances, top_mfs

def top_K_tweets(cur, embedder, embeds, theme, theme_id):
    # Get good centroids
    cur.execute('select text, mf, stance from phrase where goodness = "good" and theme_id = {}'.format(theme_id))
    phrases = cur.fetchall()
    mfs = [ph[1] for ph in phrases]
    stances = [ph[2] for ph in phrases]
    phrases = [ph[0] for ph in phrases]
    return _top_K(phrases, mfs, stances, embedder, embeds, 100)

def top_bad_K_tweets(cur, embedder, embeds, theme, theme_id):
    cur.execute('select text, mf, stance from phrase where goodness = "bad" and theme_id = {}'.format(theme_id))
    phrases = cur.fetchall()
    mfs = [ph[1] for ph in phrases]
    stances = [ph[2] for ph in phrases]
    phrases = [ph[0] for ph in phrases]
    #print(phrases)
    #exit()
    return _top_K(phrases, mfs, stances, embedder, embeds, 100)

def main(args):
    embedder = SentenceTransformer('all-mpnet-base-v2')
    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    with open(args.id2pos) as fp:
        id2pos = json.load(fp)
    pos2id = {}
    for _id in id2pos:
        positions = id2pos[_id]
        for pos in positions:
            pos2id[pos] = _id

    curr_texts = np.load(args.text)
    curr_embeds = np.load(args.sbert)

    # Get all themes
    cur.execute('select id, name from theme')
    themes =  cur.fetchall()
    print(themes)
    with open(os.path.join(args.drail_data, 'themes.txt'), "w") as fp:
        for theme in themes:
            if not theme[1].startswith('KMeans'):
                fp.write(theme[1])
                fp.write('\n')

    fp_hastheme = open(os.path.join(args.drail_data, 'has_theme.txt'), 'w')
    fp_iscand = open(os.path.join(args.drail_data, 'is_candidate.txt'), 'w')
    fp_tweet = open(os.path.join(args.drail_data, 'is_tweet.txt'), 'w')
    fp_stance = open(os.path.join(args.drail_data, 'has_stance.txt'), 'w')
    fp_morality = open(os.path.join(args.drail_data, 'has_morality.txt'), 'w')
    fp_mf = open(os.path.join(args.drail_data, 'has_mf.txt'), 'w')

    positive_examples = {}; seen_tweets = set()
    for (_id, _theme) in themes:
        #cur.execute('select tweet_id, text, distance from tweet where theme_id = {} order by distance asc limit 100'.format(_id))
        #tweets = cur.fetchall()
        #print(_theme, len(tweets))

        if not _theme.startswith('KMeans') and _theme != 'Unknown':
            positive_examples[_theme] = []
            # get good ones
            tweets, stances, mfs = top_K_tweets(cur, embedder, curr_embeds, _theme, _id)
            for tw, stance, mf in zip(tweets, stances, mfs):
                fp_hastheme.write('{}\t{}\n'.format(pos2id[tw.item()], _theme))
                fp_iscand.write('{}\t{}\n'.format(pos2id[tw.item()], _theme))

                if tw.item() not in seen_tweets:
                    fp_tweet.write('{}\n'.format(pos2id[tw.item()]))
                    fp_stance.write('{}\t{}\n'.format(pos2id[tw.item()], stance))
                    fp_mf.write('{}\t{}\n'.format(pos2id[tw.item()], mf))

                    morality = 'moral'
                    if mf == 'none':
                        morality = 'non-moral'
                    fp_morality.write('{}\t{}\n'.format(pos2id[tw.item()], morality))

                    seen_tweets.add(pos2id[tw.item()])

                positive_examples[_theme].append(pos2id[tw.item()])

            # get bad ones
            tweets, stances, mfs = top_bad_K_tweets(cur, embedder, curr_embeds, _theme, _id)
            for tw, stance, mf in zip(tweets, stances, mfs):
                fp_iscand.write('{}\t{}\n'.format(pos2id[tw.item()], _theme))

                if tw.item() not in seen_tweets:
                    fp_tweet.write('{}\n'.format(pos2id[tw.item()]))
                    fp_stance.write('{}\t{}\n'.format(pos2id[tw.item()], stance))
                    fp_mf.write('{}\t{}\n'.format(pos2id[tw.item()], mf))

                    morality = 'moral'
                    if mf == 'none':
                        morality = 'non-moral'
                    fp_morality.write('{}\t{}\n'.format(pos2id[tw.item()], morality))

                    seen_tweets.add(pos2id[tw.item()])


    # Expand negative candidates with other themes
    for _theme in positive_examples:
        for _other_theme in positive_examples:
            if _theme != _other_theme:
                for tw_id in positive_examples[_other_theme]:
                    fp_iscand.write('{}\t{}\n'.format(tw_id, _theme))

    # Expand negative candidates with random samples of K-means
    #for (_id, _theme) in themes:
    #    if _theme.startswith('KMeans'):
    #        cur.execute('select tweet_id, text, distance from tweet where theme_id = {} order by distance desc limit 100'.format(_id))
    #        tweets = cur.fetchall()
    #        for _other_theme in positive_examples:
    #            for tw in tweets:
    #                fp_iscand.write('{}\t{}\n'.format(pos2id[tw[0]], _other_theme))

    fp_hastheme.close()
    fp_iscand.close()

    tweets = cur.execute('select distinct tweet_id, stance, mf, morality from tweet')
    for (_id, stance, mf, morality) in tweets:
        tweet_id = pos2id[_id]
        if tweet_id not in seen_tweets:
            fp_tweet.write('{}\n'.format(tweet_id))
            fp_stance.write('{}\t{}\n'.format(tweet_id, stance))
            fp_morality.write('{}\t{}\n'.format(tweet_id, morality))
            fp_mf.write('{}\t{}\n'.format(tweet_id, mf))
    fp_stance.close()
    fp_morality.close()
    fp_mf.close()
    fp_tweet.close()

    # Here tweet id is the internal ID and not the POS ID, so gotta do -1
    entities = cur.execute('select entity_id, entity_text, tweet_id, sentiment from tweet__has__entity')
    fp_entity = open(os.path.join(args.drail_data, 'has_entity.txt'), 'w')
    fp_role = open(os.path.join(args.drail_data, 'has_role.txt'), 'w')
    fp_sentiment = open(os.path.join(args.drail_data, 'has_sentiment.txt'), 'w')
    for entity_id, entity_text, tweet_id, sent in tweets:
        tweet_id = pos2id[tweet_id-1]
        (role, sent) = sent.split('-')
        fp_entity.write('{}\t{}\n'.format(tweet_id, entity_id))
        fp_role.write('{}\t{}\t{}\n'.format(tweet_id, entity_id, role))
        fp_sentiment.write('{}\t{}\t{}\n'.format(tweet_id, entity_id, sent))
    fp_entity.close()
    fp_role.close()
    fp_sentiment.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--drail_data', type=str, required=True)
    parser.add_argument('--id2pos', type=str, default='flask-gui/static/files/id2pos.json')
    parser.add_argument('--sbert', type=str, default='flask-gui/static/files/sbert.npy')
    parser.add_argument('--text', type=str, default='flask-gui/static/files/text.npy')
    args = parser.parse_args()
    main(args)
