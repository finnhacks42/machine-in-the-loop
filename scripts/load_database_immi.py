import argparse
import json
import sqlite3
from tqdm import tqdm

def parse_predicate(pred):
    name, args = pred.split('(')
    args = args[:-1].split(',')
    return name, args

def main(args):
    with open(args.id2pos) as fp:
        id2pos = json.load(fp)

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    if args.init:
        # First time updating from json file
        with open('flask-gui/static/immi/files/id2explanations.json') as fp:
            id2explanations = json.load(fp)

        pbar = tqdm(total=len(id2explanations), desc='dumping explanations...')
        for tweet_id in id2explanations:
            frame_political = False
            frame_policy = False
            frame_crime = False
            frame_health = False
            frame_security = False
            frame_economic = False

            for frame in id2explanations[tweet_id]['policy_frame']:
                if frame == 'Political Factors and Implications':
                    frame_political = True
                if frame == 'Policy Prescription and Evaluation':
                    frame_policy = True
                if frame == 'Crime and Punishment':
                    frame_crime = True
                if frame == 'Health and Safety':
                    frame_health = True
                if frame == 'Security and Defense':
                    frame_security = True
                if frame == 'Economic':
                    frame_economic = True

            positions = id2pos[tweet_id]

            for pos in positions:
                query = 'UPDATE tweet_immi SET \
                            narrative = "{0}", \
                            immi_frame = "{1}", \
                            immi_role = "{2}", \
                            frame_political = {3}, \
                            frame_policy = {4}, \
                            frame_crime = {5}, \
                            frame_health = {6}, \
                            frame_security = {7}, \
                            frame_economic = {8} \
                        WHERE tweet_id = {9}'
                query = query.format(
                             id2explanations[tweet_id]['narrative'],
                             id2explanations[tweet_id]['immi_frame'],
                             id2explanations[tweet_id]['role'],
                             frame_political,
                             frame_policy,
                             frame_crime,
                             frame_health,
                             frame_security,
                             frame_economic,
                             pos)
                cur.execute(query)
            pbar.update(1)
        pbar.close()
    else:
        tweet2value = {}
        with open(args.input_preds) as fp:
            lines = fp.readlines()
            pbar = tqdm(total=len(lines), desc='reading predictions...')

            for line in lines:
                name, args = parse_predicate(line.strip())
                if name == "HasNarrative":
                    tweet_id = args[0]
                    narrative = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    tweet2value[tweet_id]['narrative'] = narrative
                elif name == "HasImmiFrame":
                    tweet_id = args[0]
                    immi_frame = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    tweet2value[tweet_id]['immi_frame'] = immi_frame
                elif name == "HasImmiRole":
                    tweet_id = args[0]
                    immi_role = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    tweet2value[tweet_id]['immi_role'] = immi_role
                elif name == "HasFramePolitical":
                    tweet_id = args[0]
                    value = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    if int(value) == 1:
                        tweet2value[tweet_id]['frame_political'] = True
                    else:
                        tweet2value[tweet_id]['frame_political'] = False
                elif name == "HasFramePolicy":
                    tweet_id = args[0]
                    value = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    if int(value) == 1:
                        tweet2value[tweet_id]['frame_policy'] = True
                    else:
                        tweet2value[tweet_id]['frame_policy'] = False
                elif name == "HasFrameCrime":
                    tweet_id = args[0]
                    value = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    if int(value) == 1:
                        tweet2value[tweet_id]['frame_crime'] = True
                    else:
                        tweet2value[tweet_id]['frame_crime'] = False
                elif name == "HasFrameHealth":
                    tweet_id = args[0]
                    value = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    if int(value) == 1:
                        tweet2value[tweet_id]['frame_health'] = True
                    else:
                        tweet2value[tweet_id]['frame_health'] = False
                elif name == "HasFrameSecurity":
                    tweet_id = args[0]
                    value = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    if int(value) == 1:
                        tweet2value[tweet_id]['frame_security'] = True
                    else:
                        tweet2value[tweet_id]['frame_security'] = False
                elif name == "HasFrameEconomic":
                    tweet_id = args[0]
                    value = args[-1]
                    if tweet_id not in tweet2value:
                        tweet2value[tweet_id] = {}
                    if int(value) == 1:
                        tweet2value[tweet_id]['frame_economic'] = True
                    else:
                        tweet2value[tweet_id]['frame_economic'] = False

                pbar.update(1)
            pbar.close()

        pbar = tqdm(total=len(tweet2value), desc='updating dataset')
        for tweet_id in tweet2value:
            positions = id2pos[tweet_id]
            for pos in positions:
                query = 'UPDATE tweet_immi SET \
                            narrative = "{0}", \
                            immi_frame = "{1}", \
                            immi_role = "{2}", \
                            frame_political = {3}, \
                            frame_policy = {4}, \
                            frame_crime = {5}, \
                            frame_health = {6}, \
                            frame_security = {7}, \
                            frame_economic = {8} \
                        WHERE tweet_id = {9}'
                query = query.format(
                             tweet2value[tweet_id]['narrative'],
                             tweet2value[tweet_id]['immi_frame'],
                             tweet2value[tweet_id]['immi_role'],
                             tweet2value[tweet_id]['frame_political'],
                             tweet2value[tweet_id]['frame_policy'],
                             tweet2value[tweet_id]['frame_crime'],
                             tweet2value[tweet_id]['frame_health'],
                             tweet2value[tweet_id]['frame_security'],
                             tweet2value[tweet_id]['frame_economic'],
                             pos)
                cur.execute(query)
            pbar.update(1)
        pbar.close()

    conn.commit()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', default=False, action='store_true')
    parser.add_argument('--input_preds', type=str, required=True)
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--id2pos', type=str, required=True)
    args = parser.parse_args()
    main(args)
