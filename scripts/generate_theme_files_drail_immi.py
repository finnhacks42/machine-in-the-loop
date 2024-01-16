import argparse
import sqlite3
import json
import numpy as np
from tqdm import tqdm
import os
import torch
from sentence_transformers import SentenceTransformer, util

def _top_K(phrases, narratives, immi_frames, immi_roles,
           frame_political, frame_policy, frame_crime,
           frame_health, frame_security, frame_economic,
           embedder, embeds, K):
    top_results = []; 
    top_narratives = []; top_immi_frames = []; top_immi_roles = []
    top_frame_political = []; top_frame_policy = []; top_frame_crime = []
    top_frame_health = []; top_frame_security = []; top_frame_economic = []

    if len(phrases) > 0:
        centroids = embedder.encode(phrases)

        centroid_distances = [];

        for centroid in centroids:

            cos_scores = util.cos_sim(centroid, embeds)[0]
            centroid_distances.append(cos_scores)

        out = torch.column_stack(centroid_distances)
        out_max = torch.max(out, 1)

        max_cos_scores = out_max.values
        max_cos_indices = out_max.indices


        res_narratives = [narratives[idx] for idx in max_cos_indices]
        res_immi_frames = [immi_frames[idx] for idx in max_cos_indices]
        res_immi_roles = [immi_roles[idx] for idx in max_cos_indices]

        res_frame_political = [frame_political[idx] for idx in max_cos_indices]
        res_frame_policy = [frame_policy[idx] for idx in max_cos_indices]
        res_frame_crime = [frame_crime[idx] for idx in max_cos_indices]
        res_frame_health = [frame_health[idx] for idx in max_cos_indices]
        res_frame_security = [frame_security[idx] for idx in max_cos_indices]
        res_frame_economic = [frame_economic[idx] for idx in max_cos_indices]

        print(max_cos_scores.shape)

        top_results = torch.topk(max_cos_scores, k=K).indices
        top_scores = max_cos_scores[top_results]
        print(top_scores)
        print(top_scores.shape)

        top_narratives = [res_narratives[idx] for idx in top_results]
        top_immi_frames = [res_immi_frames[idx] for idx in top_results]
        top_immi_roles = [res_immi_roles[idx] for idx in top_results]

        top_frame_political = [res_frame_political[idx] for idx in top_results]
        top_frame_policy = [res_frame_policy[idx] for idx in top_results]
        top_frame_crime = [res_frame_crime[idx] for idx in top_results]
        top_frame_health= [res_frame_health[idx] for idx in top_results]
        top_frame_security = [res_frame_security[idx] for idx in top_results]
        top_frame_economic = [res_frame_economic[idx] for idx in top_results]


    return top_results, top_narratives, top_immi_frames, top_immi_roles,\
           top_frame_political, top_frame_policy, top_frame_crime,\
           top_frame_health, top_frame_security, top_frame_economic

def top_K_tweets(cur, embedder, embeds, theme, theme_id):
    # Get good centroids
    cur.execute('select text, narrative, immi_frame, immi_role, frame_political, frame_policy, frame_crime, frame_health, frame_security, frame_economic from phrase_immi where goodness = "good" and theme_id = {}'.format(theme_id))
    phrases = cur.fetchall()

    narratives = [ph[1] for ph in phrases]
    immi_frames = [ph[2] for ph in phrases]
    immi_roles = [ph[3] for ph in phrases]

    frame_political = [ph[4] for ph in phrases]
    frame_policy = [ph[5] for ph in phrases]
    frame_crime = [ph[6] for ph in phrases]
    frame_health = [ph[7] for ph in phrases]
    frame_security = [ph[8] for ph in phrases]
    frame_economic = [ph[9] for ph in phrases]

    phrases = [ph[0] for ph in phrases]
    print(theme)
    return _top_K(phrases, narratives, immi_frames, immi_roles,
                  frame_political, frame_policy, frame_crime,
                  frame_health, frame_security, frame_economic,
                  embedder, embeds, 100)

def top_bad_K_tweets(cur, embedder, embeds, theme, theme_id):
    cur.execute('select text, narrative, immi_frame, immi_role, frame_political, frame_policy, frame_crime, frame_health, frame_security, frame_economic from phrase_immi where goodness = "bad" and theme_id = {}'.format(theme_id))
    phrases = cur.fetchall()

    narratives = [ph[1] for ph in phrases]
    immi_frames = [ph[2] for ph in phrases]
    immi_roles = [ph[3] for ph in phrases]

    frame_political = [ph[4] for ph in phrases]
    frame_policy = [ph[5] for ph in phrases]
    frame_crime = [ph[6] for ph in phrases]
    frame_health = [ph[7] for ph in phrases]
    frame_security = [ph[8] for ph in phrases]
    frame_economic = [ph[9] for ph in phrases]

    phrases = [ph[0] for ph in phrases]

    return _top_K(phrases, narratives, immi_frames, immi_roles,
                  frame_political, frame_policy, frame_crime,
                  frame_health, frame_security, frame_economic,
                  embedder, embeds, 100)

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
            if not theme[1].startswith('KMeans') and not theme[1] == "Unknown":
                fp.write(theme[1])
                fp.write('\n')

    fp_hastheme = open(os.path.join(args.drail_data, 'has_theme.txt'), 'w')
    fp_iscand = open(os.path.join(args.drail_data, 'is_candidate.txt'), 'w')
    fp_tweet = open(os.path.join(args.drail_data, 'is_tweet.txt'), 'w')

    fp_narrative = open(os.path.join(args.drail_data, 'has_narrative.txt'), 'w')
    fp_immi_frame = open(os.path.join(args.drail_data, 'has_immi_frame.txt'), 'w')
    fp_immi_role = open(os.path.join(args.drail_data, 'has_immi_role.txt'), 'w')

    fp_frame_political = open(os.path.join(args.drail_data, 'has_frame_political.txt'), 'w')
    fp_frame_policy = open(os.path.join(args.drail_data, 'has_frame_policy.txt'), 'w')
    fp_frame_crime = open(os.path.join(args.drail_data, 'has_frame_crime.txt'), 'w')
    fp_frame_health = open(os.path.join(args.drail_data, 'has_frame_health.txt'), 'w')
    fp_frame_security = open(os.path.join(args.drail_data, 'has_frame_security.txt'), 'w')
    fp_frame_economic = open(os.path.join(args.drail_data, 'has_frame_economic.txt'), 'w')

    positive_examples = {}; seen_tweets = set()
    for (_id, _theme) in themes:
        #cur.execute('select tweet_id, text, distance from tweet where theme_id = {} order by distance asc limit 100'.format(_id))
        #tweets = cur.fetchall()
        #print(_theme, len(tweets))

        if not _theme.startswith('KMeans') and _theme != 'Unknown':
            positive_examples[_theme] = []
            # get good ones
            (tweets, top_narratives, top_immi_frames, top_immi_roles,\
                top_frame_political, top_frame_policy, top_frame_crime,\
                top_frame_health, top_frame_security, top_frame_economic)\
                    = top_K_tweets(cur, embedder, curr_embeds, _theme, _id)

            for i in range(0, len(tweets)):

                tw = tweets[i]
                fp_hastheme.write('{}\t{}\n'.format(pos2id[tw.item()], _theme))
                fp_iscand.write('{}\t{}\n'.format(pos2id[tw.item()], _theme))

                if pos2id[tw.item()] not in seen_tweets:
                    fp_tweet.write('{}\n'.format(pos2id[tw.item()]))

                    fp_narrative.write('{}\t{}\n'.format(pos2id[tw.item()], top_narratives[i]))
                    fp_immi_frame.write('{}\t{}\n'.format(pos2id[tw.item()], top_immi_frames[i]).replace(' ', '_').replace(':', ''))
                    fp_immi_role.write('{}\t{}\n'.format(pos2id[tw.item()], top_immi_roles[i]))

                    fp_frame_political.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_political[i]))
                    fp_frame_policy.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_policy[i]))
                    fp_frame_crime.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_crime[i]))
                    fp_frame_health.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_health[i]))
                    fp_frame_security.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_security[i]))
                    fp_frame_economic.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_economic[i]))

                    seen_tweets.add(pos2id[tw.item()])

                positive_examples[_theme].append(pos2id[tw.item()])

            # get bad ones
            (tweets, top_narratives, top_immi_frames, top_immi_roles,\
                top_frame_political, top_frame_policy, top_frame_crime,\
                top_frame_health, top_frame_security, top_frame_economic)\
                    = top_bad_K_tweets(cur, embedder, curr_embeds, _theme, _id)

            for i in range(0, len(tweets)):
                tw = tweets[i]
                fp_iscand.write('{}\t{}\n'.format(pos2id[tw.item()], _theme))

                if pos2id[tw.item()] not in seen_tweets:
                    fp_tweet.write('{}\n'.format(pos2id[tw.item()]))

                    fp_narrative.write('{}\t{}\n'.format(pos2id[tw.item()], top_narratives[i]))
                    fp_immi_frame.write('{}\t{}\n'.format(pos2id[tw.item()], top_immi_frames[i].replace(' ', '_').replace(':', '')))
                    fp_immi_role.write('{}\t{}\n'.format(pos2id[tw.item()], top_immi_roles[i]))

                    fp_frame_political.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_political[i]))
                    fp_frame_policy.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_policy[i]))
                    fp_frame_crime.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_crime[i]))
                    fp_frame_health.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_health[i]))
                    fp_frame_security.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_security[i]))
                    fp_frame_economic.write('{}\t{}\n'.format(pos2id[tw.item()], top_frame_economic[i]))

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

    tweets = cur.execute('select distinct tweet_id, narrative, immi_frame, immi_role, frame_political, frame_policy, frame_crime, frame_health, frame_security, frame_economic from tweet_immi').fetchall()
    for i in range(0, len(tweets)):
        _id = tweets[i][0]
        tweet_id = pos2id[_id]
        if tweet_id not in seen_tweets:
            fp_tweet.write('{}\n'.format(tweet_id))

            fp_narrative.write('{}\t{}\n'.format(tweet_id, tweets[i][1]))
            fp_immi_frame.write('{}\t{}\n'.format(tweet_id, tweets[i][2].replace(' ', '_').replace(':', '')))
            fp_immi_role.write('{}\t{}\n'.format(tweet_id, tweets[i][3]))

            fp_frame_political.write('{}\t{}\n'.format(tweet_id, tweets[i][4]))
            fp_frame_policy.write('{}\t{}\n'.format(tweet_id, tweets[i][5]))
            fp_frame_crime.write('{}\t{}\n'.format(tweet_id, tweets[i][6]))
            fp_frame_health.write('{}\t{}\n'.format(tweet_id, tweets[i][7]))
            fp_frame_security.write('{}\t{}\n'.format(tweet_id, tweets[i][8]))
            fp_frame_economic.write('{}\t{}\n'.format(tweet_id, tweets[i][9]))

            seen_tweets.add(tweet_id)

    fp_tweet.close()
    fp_narrative.close()
    fp_immi_frame.close()
    fp_immi_role.close()

    fp_frame_political.close()
    fp_frame_policy.close()
    fp_frame_crime.close()
    fp_frame_health.close()
    fp_frame_security.close()
    fp_frame_economic.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--drail_data', type=str, required=True)
    parser.add_argument('--id2pos', type=str, default='flask-gui/static/files/id2pos.json')
    parser.add_argument('--sbert', type=str, default='flask-gui/static/files/sbert.npy')
    parser.add_argument('--text', type=str, default='flask-gui/static/files/text.npy')
    args = parser.parse_args()
    main(args)
