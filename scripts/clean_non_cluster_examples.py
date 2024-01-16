import argparse
import sqlite3
import json
import numpy as np
from tqdm import tqdm

def parse_predicate(pred):
    name, args = pred.split('(')
    args = args[:-1].split(',')
    return name, args

def main(args):
    print(args)
    valid_tweets = set(); all_tweets = set()
    with open(args.input_preds) as fp:
        for line in fp:
            name, argums = parse_predicate(line.strip())
            if name == 'MentionsArgument':
                valid_tweets.add(argums[0])
            else:
                all_tweets.add(argums[0])
    print(len(valid_tweets))
    print(len(all_tweets))

    fp_out = open(args.output_preds, 'w')
    with open(args.input_preds) as fp:
        for line in fp:
            name, argums = parse_predicate(line.strip())

            if argums[0] in valid_tweets:
                fp_out.write(line)
    fp_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_preds', type=str, required=True)
    parser.add_argument('--output_preds', type=str, required=True)
    args = parser.parse_args()
    main(args)

