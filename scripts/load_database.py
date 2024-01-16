import argparse
import sqlite3
import json
import numpy as np
from tqdm import tqdm

def parse_predicate(pred):
    name, args = pred.split('(')
    args = args[:-1].split(',')
    return name, args

def map_stances(stance_idx):
    if stance_idx == 0:
        return 'pro-vax'
    elif stance_idx == 1:
        return 'anti-vax'
    else:
        return 'neutral'

def main(args):
    with open(args.id2text) as fp:
        id2text = json.load(fp)
    with open(args.id2pos) as fp:
        id2pos = json.load(fp)
    with open(args.entity2lexform) as fp:
        entity2lexform = json.load(fp)

    curr_texts = np.load(args.text)
    curr_texts = list(curr_texts)

    '''
    for tw in id2text:
        tw_text = id2text[tw]
        indices = [i for i,d in enumerate(curr_texts) if d==tw_text]
        id2pos[tw] = indices
    with open(args.id2pos, 'w') as fp:
        json.dump(id2pos, fp)
    '''


    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    for colname in ['stance', 'morality', 'mf']:
        try:
            cur.execute('alter table tweet add column {} TEXT'.format(colname))
        except:
            print('column "{}" already exists'.format(colname))

    try:
        cur.execute('drop table tweet__has__entity')
    except:
        pass
    cur.execute('create table if not exists tweet__has__entity \
                (id INTEGER PRIMARY KEY AUTOINCREMENT, \
                 tweet_id INTEGER NOT NULL, \
                 entity_id INTEGER NOT NULL, \
                 entity_text TEXT NOT NULL, \
                 sentiment TEXT NOT NULL, \
                 FOREIGN KEY(tweet_id) REFERENCES tweet(id))')

    with open(args.input_preds) as fp:
        lines = fp.readlines()
        pbar = tqdm(total=len(lines), desc='reading predictions...')

        tweet2symbols = {}; entity2symbols = {}
        for line in lines:
            name, args = parse_predicate(line.strip())

            if name == 'HasRole':
                role = args[2]
                ent = args[1]
                tweet = args[0]
                lexform = entity2lexform[ent].lower().replace('"', "'")

                if ent not in entity2symbols:
                    entity2symbols[ent] = {}
                    entity2symbols[ent]['lexform'] = lexform
                    entity2symbols[ent]['tweet'] = tweet
                entity2symbols[ent]['role'] = role

            elif name == 'HasSentiment':
                sent = args[2]
                ent = args[1]
                tweet = args[0]
                lexform = entity2lexform[ent].lower().replace('"', "'")

                if ent not in entity2symbols:
                    entity2symbols[ent] = {}
                    entity2symbols[ent]['lexform'] = lexform
                    entity2symbols[ent]['tweet'] = tweet
                entity2symbols[ent]['sent'] = sent


            elif name == 'HasMoralFoundation':
                mf = args[1]
                tweet = args[0]
                if tweet not in tweet2symbols:
                    tweet2symbols[tweet] = {}
                tweet2symbols[tweet]['mf'] = mf
            elif name == 'HasMorality':
                morality = args[1]
                tweet = args[0]
                if tweet not in tweet2symbols:
                    tweet2symbols[tweet] = {}
                tweet2symbols[tweet]['morality'] = morality
            elif name == 'HasStance':
                stance = args[1]
                tweet = args[0]
                if tweet not in tweet2symbols:
                    tweet2symbols[tweet] = {}
                if stance in ['0', '1', '2']:
                    stance = map_stances(int(stance))
                tweet2symbols[tweet]['stance'] = stance
            else:
                continue
            pbar.update(1)
        pbar.close()

        pbar = tqdm(total=len(entity2symbols), desc='inserting entity rows')
        for ent in entity2symbols:
            tweet = entity2symbols[ent]['tweet']
            positions = id2pos[tweet]
            for pos in positions:
                lexform = entity2symbols[ent]['lexform']
                role = entity2symbols[ent]['role']
                sent = entity2symbols[ent]['sent']
                sent = role + "-" + sent[-3:]

                query = 'INSERT INTO tweet__has__entity (tweet_id, entity_id, entity_text, sentiment) VALUES ({}, {}, "{}", "{}")'
                query = query.format(int(pos) + 1, ent, lexform.lower(), sent)
                #print(query)
                cur.execute(query)
            pbar.update(1)
        pbar.close()

        pbar = tqdm(total=len(tweet2symbols), desc='updating dataset')
        for tweet in tweet2symbols:
            #print(tweet2symbols[tweet])
            positions = id2pos[tweet]
            for pos in positions:
                #Sanity check here

                #text_1 = id2text[tweet]
                #text_2 = curr_texts[pos]
                #print('----')
                #print(text_1)
                #print('----')
                #print(text_2)
                #print('/////')
                #exit()

                query = 'UPDATE tweet SET morality = "{0}", stance = "{1}", mf = "{2}" WHERE tweet_id = {3}'
                query = query.format(tweet2symbols[tweet]['morality'],
                                     tweet2symbols[tweet]['stance'],
                                     tweet2symbols[tweet]['mf'],
                                     pos)
                #print(query)
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
    parser.add_argument('--id2text', type=str, required=True)
    parser.add_argument('--entity2lexform', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    main(args)
