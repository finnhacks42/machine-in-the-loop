import json
import logging.config
import argparse
import numpy as np
from sklearn.metrics import *
import torch
import random
from pathlib import Path
import os
from drail import database
from drail.learn.local_learner import LocalLearner
from termcolor import colored

import warnings
warnings.filterwarnings('ignore') 

def create_folds(data_folder):
    tweets = []
    with open(os.path.join(data_folder, 'is_candidate.txt')) as fp:
        for line in fp:
            tweet, _ = line.strip().split()
            tweets.append(tweet)
    tweets = list(set(tweets))
    random.shuffle(tweets)
    n_train = int(len(tweets) * 0.8)
    train_tweets = tweets[:n_train]
    dev_tweets = tweets[n_train:]

    test_tweets = []
    with open(os.path.join(data_folder, 'is_tweet.txt')) as fp:
        for line in fp:
            test_tweets.append(line.strip())
    #test_tweets = test_tweets[:100]

    return train_tweets, dev_tweets, test_tweets

def process_head(head):
    name, args = head.split('(')
    args = args[:-1]
    args = args.split(',')
    return name, args

def seed_examples(args, dataset, seed_tweet_ids):
    for predicate in dataset:
        with open(os.path.join(args.dir, predicate + ".txt")) as fp:
            for line in fp:
                elems = line.strip().split('\t')
                if int(elems[0]) in seed_tweet_ids:
                    dataset[predicate].add(tuple(elems))

def expectation_step(learner, db, args, dev_tweets):
    dataset = {
        'has_mf': set(),
        'has_morality': set(),
        'has_role': set(),
        'has_sentiment': set(),
        'has_stance': set()
    }

    learner.init_models()

    learner.extract_instances(db,
            extract_train=True, test_filters=[("IsTweet", "isTrain", 1)])
    res, heads = learner.predict(db, fold_filters=[("IsTweet", "isTrain", 1)], fold='train', get_predicates=True)
    # Now write the updated files
    for h in heads:
        name, argums = process_head(h)
        tw_id = argums[0]
        if name == 'HasMoralFoundation':
            dataset['has_mf'].add(tuple(argums))
        elif name == 'HasMorality':
            dataset['has_morality'].add(tuple(argums))
        elif name == 'HasRole':
            dataset['has_role'].add(tuple(argums))
        elif name == 'HasSentiment':
            dataset['has_sentiment'].add(tuple(argums))
        elif name == 'HasStance':
            dataset['has_stance'].add(tuple(argums))

    seed_examples(args, dataset, dev_tweets)

    for predicate in dataset:
        with open(os.path.join(args.dir_dyn, predicate + '.txt'), 'w') as fp:
            for tup in dataset[predicate]:
                fp.write('\t'.join(tup))
                fp.write('\n')

    learner.reset_metrics()
    db = learner.create_dataset(args.dir_dyn)
    return db

def maximization_step(learner, db, optimizer):
    learner.train(db,
                  train_filters=[("IsTweet", "isTrain", 1)],
                  dev_filters=[("IsTweet", "isDev", 1)],
                  test_filters=[("IsTweet", "isTest", 1)],
                  optimizer=optimizer)

def evaluate_dev(learner, db, filter_name, fold_name):
    learner.init_models()
    learner.extract_instances(db,
            extract_dev=True, test_filters=[("IsTweet", filter_name, 1)])
    res, heads = learner.predict(db, fold_filters=[("IsTweet", filter_name, 1)], fold=fold_name, get_predicates=True)
    weighted_f1 = 0; n_count = 1
    for pred in set(['HasRole', 'HasMoralFoundation', 'HasSentiment', 'HasStance', 'HasMorality']):
        #if pred in set(['HasMoralFoundation']):
            y_gold = res.metrics[pred]['gold_data']
            y_pred = res.metrics[pred]['pred_data']
            weighted_f1 += f1_score(y_gold, y_pred, average='weighted')
            n_count += 1
            #print(classification_report(y_gold, y_pred, digits=4))
    learner.reset_metrics()
    return weighted_f1/n_count

    for pred in res.metrics:
        if pred in set(['HasRole', 'HasMoralFoundation', 'HasSentiment', 'HasStance', 'HasMorality']):
            y_gold = res.metrics[pred]['gold_data']
            y_pred = res.metrics[pred]['pred_data']
            if pred not in avg_metrics:
                avg_metrics[pred] = {}
                avg_metrics[pred]['gold'] = y_gold
                avg_metrics[pred]['pred'] = y_pred
            else:
                avg_metrics[pred]['gold'].extend(y_gold)
                avg_metrics[pred]['pred'].extend(y_pred)

            #logger.info('\n'+ pred + ':\n' + classification_report(y_gold, y_pred, digits=4))
    learner.reset_metrics()

def main(args):
    optimizer = "AdamW"
    if args.gpu_index:
        torch.cuda.set_device(args.gpu_index)

    learner = LocalLearner()
    learner.compile_rules(args.rules)
    db = learner.create_dataset(args.dir)

    torch.cuda.empty_cache()
    curr_savedir = args.savedir
    Path(curr_savedir).mkdir(parents=True, exist_ok=True)

    train_tweets, dev_tweets, test_tweets = create_folds(args.dir)
    print("Train", len(train_tweets), "Dev", len(dev_tweets), "Test", len(test_tweets))

    if args.do_train:

        db.add_filters(filters=[
            ("IsTweet", "isDummy", "tweetId_1", train_tweets[0:1]),
            ("IsTweet", "isTrain", "tweetId_1", train_tweets),
            ("IsTweet", "isDev", "tweetId_1", dev_tweets),
            ("IsTweet", "isTest", "tweetId_1", dev_tweets)
        ])

        learner.build_feature_extractors(db,
                tweets_f=args.tweet2bert,
                entities_f=args.entity2bert,
                themes_f=os.path.join(args.dir, 'themes.txt'),
                femodule_path='drail_programs/',
                filters=[("IsTweet", "isDummy", 1)])

        learner.set_savedir(curr_savedir)
        learner.build_models(db, args.config, netmodules_path='drail_programs/')
        learner.init_models()

        best_f1_dev = evaluate_dev(learner, db, 'isDev', 'dev')
        term = "Dev: {}".format(best_f1_dev)
        print(colored(term, 'red'))

        PAT = 2
        patience = PAT

        n_epochs = 0
        while (patience > 0):

            db = expectation_step(learner, db, args, dev_tweets)

            db.add_filters(filters=[
                ("IsTweet", "isDummy", "tweetId_1", train_tweets[0:1]),
                ("IsTweet", "isTrain", "tweetId_1", train_tweets),
                ("IsTweet", "isDev", "tweetId_1", dev_tweets),
                ("IsTweet", "isTest", "tweetId_1", dev_tweets)
            ])

            maximization_step(learner, db, optimizer)
            f1_dev = evaluate_dev(learner, db, 'isDev', 'dev')

            if f1_dev > best_f1_dev:
                best_f1_dev = f1_dev
                patience = PAT
            else:
                patience -= 1

            n_epochs += 1
            term = "Epoch: {}, Dev: {}".format(n_epochs, best_f1_dev)
            print(colored(term, 'red'))

    if args.do_predict:
        db.add_filters(filters=[
            ("IsTweet", "isDummy", "tweetId_1", train_tweets[0:1]),
            ("IsTweet", "isTrain", "tweetId_1", train_tweets[0:1]),
            ("IsTweet", "isDev", "tweetId_1", train_tweets[0:1]),
            ("IsTweet", "isTest", "tweetId_1", test_tweets)
        ])

        learner.build_feature_extractors(db,
                tweets_f=args.tweet2bert,
                entities_f=args.entity2bert,
                themes_f=os.path.join(args.dir, 'themes.txt'),
                femodule_path='drail_programs/',
                filters=[("IsTweet", "isDummy", 1)])

        learner.set_savedir(curr_savedir)
        learner.build_models(db, args.config, netmodules_path='drail_programs/')
        learner.init_models()
        learner.extract_instances(db,
                extract_test=True, test_filters=[("IsTweet", "isTest", 1)])
        res, heads = learner.predict(db, fold_filters=[("IsTweet", "isTest", 1)], fold='test', get_predicates=True)
        with open(args.out_dir, "w") as fp:
            for h in heads:
                fp.write(str(h))
                fp.write("\n")

if __name__ == "__main__":
    # seed and cuda
    torch.manual_seed(1534)
    np.random.seed(1534)
    random.seed(1534)

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', help='gpu index', dest='gpu_index', type=int, default=None)
    parser.add_argument('-d', '--dir', help='directory', dest='dir', type=str, required=True)
    parser.add_argument('--dir_dyn', help='dynamic directory', dest='dir_dyn', type=str, required=True)
    parser.add_argument('-r', '--rule', help='rule file', dest='rules', type=str, required=True)
    parser.add_argument('-c', '--config', help='config file', dest='config', type=str, required=True)
    parser.add_argument('--savedir', help='save directory', dest='savedir', type=str, required=True)
    parser.add_argument('--tweet2bert', type=str, required=True)
    parser.add_argument('--entity2bert', type=str, required=True)
    parser.add_argument('--log', type=str, default='logging_conf.json')
    parser.add_argument('--do_train', default=False, action='store_true')
    args = parser.parse_args()

    logger = logging.getLogger()
    logging.config.dictConfig(json.load(open(args.log)))
    cmd = "rm -rf {0}".format(args.dir_dyn)
    print(cmd)
    os.system(cmd)
    cmd = "cp -r {0} {1}".format(args.dir, args.dir_dyn)
    print(cmd)
    os.system(cmd)

    main(args)

