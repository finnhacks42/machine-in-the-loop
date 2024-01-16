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
from collections import Counter

import warnings
warnings.filterwarnings('ignore') 

def create_folds(data_folder):
    test_tweets = []
    with open(os.path.join(data_folder, 'is_tweet.txt')) as fp:
        for line in fp:
            test_tweets.append(line.strip())
    #test_tweets = test_tweets[:100]

    tweets = []
    with open(os.path.join(data_folder, 'is_candidate.txt')) as fp:
        for line in fp:
            tweet, _ = line.strip().split()
            tweets.append(tweet)
    tweets = list(set(tweets))
    '''
    candidate_tweets = set(tweets)
    random.shuffle(test_tweets)
    intersection = list(set(test_tweets) - candidate_tweets)

    # Adding some additional tweets for diversity
    tweets = tweets + intersection[:1200]
    '''
    random.shuffle(tweets)
    n_train = int(len(tweets) * 0.8)
    train_tweets = tweets[:n_train]
    dev_tweets = tweets[n_train:]

    '''
    mfs = []
    with open(os.path.join(data_folder, 'has_mf.txt')) as fp:
        for line in fp:
            tweet, mf = line.strip().split()
            if tweet in tweets:
                mfs.append(mf)
    print(Counter(mfs))
    '''
    return train_tweets, dev_tweets, test_tweets

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
    if args.debug:
        train_tweets = train_tweets[:10]
        dev_tweets = dev_tweets[:10]
        test_tweets = test_tweets[:10]

    logger.info("train {}, dev {}, test {}".format(len(train_tweets), len(dev_tweets), len(test_tweets)))

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

        learner.train(db,
                      train_filters=[("IsTweet", "isTrain", 1)],
                      dev_filters=[("IsTweet", "isDev", 1)],
                      test_filters=[("IsTweet", "isTest", 1)],
                      optimizer=optimizer)

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
        learner.extract_instances(db, extract_test=True, test_filters=[("IsTweet", "isTest", 1)])
        res, heads = learner.predict(db, fold='test', get_predicates=True, fold_filters=[("IsTweet", "isTest", 1)])
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
    parser.add_argument('-g', '--gpu', help='gpu index', dest='gpu_index', type=int, default=0)
    parser.add_argument('-d', '--dir', help='directory', dest='dir', type=str, default='drail_programs/data')
    parser.add_argument('-r', '--rule', help='rule file', dest='rules', type=str, required=True)
    parser.add_argument('-c', '--config', help='config file', dest='config', type=str, required=True)
    parser.add_argument('--savedir', help='save directory', dest='savedir', type=str, required=True)
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_predict', default=False, action='store_true')
    parser.add_argument('--tweet2bert', type=str, required=True)
    parser.add_argument('--entity2bert', type=str, default=None)
    parser.add_argument('--log', type=str, default='logging_conf.json')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    logger = logging.getLogger()
    logging.config.dictConfig(json.load(open(args.log)))
    main(args)
