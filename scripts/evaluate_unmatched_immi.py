import argparse
import json
import sqlite3
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
from aggregated_distance_metrics import get_all_quartiles, get_theme_quartiles

def parse_predicate(pred):
    name, args = pred.split('(')
    args = args[:-1].split(',')
    return name, args

def main(args):
    id2pos = json.load(open(args.id2pos))

    print(args)
    pred2theme_first = {}
    pred2theme_second = {}
    all_ids = set()

    # Parse first
    themes_first = []
    with open(args.input_preds_first) as fp:
        lines = fp.readlines()
        for line in lines:
            name, _args = parse_predicate(line.strip())
            if name == "MentionsArgument":
                twid = _args[0]
                theme = _args[1]

                if theme.startswith('KMeans') or theme == 'Unknown':
                    continue
                else:
                    pred2theme_first[twid] = theme
                    themes_first.append(theme)
            elif name == "HasNarrative":
                twid = _args[0]
                all_ids.add(twid)

    print(len(all_ids))

    # Parse second
    with open(args.input_preds_second) as fp:
        lines = fp.readlines()
        for line in lines:
            name, _args = parse_predicate(line.strip())
            if name == "MentionsArgument":
                twid = _args[0]
                theme = _args[1]

                if theme.startswith('KMeans') or theme == 'Unknown':
                    continue
                else:
                    pred2theme_second[twid] = theme

    first_labels = []; second_labels = []
    all_ids = list(all_ids)
    for twid in all_ids:
        if twid in pred2theme_first:
            first_labels.append(pred2theme_first[twid])
        else:
            first_labels.append('Unknown')
        if twid in pred2theme_second:
            second_labels.append(pred2theme_second[twid])
        else:
            second_labels.append('Unknown')
    labels = np.array(list(set(first_labels) | set(second_labels)))
    just_second = set(second_labels) - set(first_labels)

    # Inspect the ones that were moved to unknown
    unmatched_ids = []
    for i, _ in enumerate(all_ids):
        if first_labels[i] != 'Unknown' and second_labels[i] == 'Unknown':
            unmatched_ids.append(all_ids[i])

    conn = sqlite3.connect(args.database_first)
    cur = conn.cursor()
    (Q1, Q2, Q3, Q4) = get_all_quartiles(cur, isimmi=True)
    print(len(Q1), len(Q2), len(Q3), len(Q4))
    # Get tw ids only
    Q1 = set([t[0] for t in Q1])
    Q2 = set([t[0] for t in Q2])
    Q3 = set([t[0] for t in Q3])
    Q4 = set([t[0] for t in Q4])

    num_q4 = 0; num_q3 = 0; num_q2 = 0; num_q1 = 0
    for twid in unmatched_ids:
        twid = id2pos[twid][0]
        if twid in Q4 and twid not in Q3:
            num_q4 += 1
        elif twid in Q3 and twid not in Q2:
            num_q3 += 1
        elif twid in Q2 and twid not in Q1:
            num_q2 += 1
        else:
            num_q1 += 1

    print("Q4", num_q4)
    print("Q3", num_q3)
    print("Q2", num_q2)
    print("Q1", num_q1)

    cm = confusion_matrix(first_labels, second_labels, labels=labels, normalize='all')
    order = np.argsort(-cm.diagonal())
    plt.figure(layout='constrained')
    cm = cm[order,:][:,order]
    cm = cm[~np.all(cm == 0, axis=1)]
    ax = sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues')
    labels = labels[order]; first_labels = [l for l in labels if l not in just_second]
    ax.set_yticklabels(first_labels, rotation=0)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('First Iteration')
    ax.set_xlabel('Second Iteration')

    # Set colors
    for ticklabel in plt.gca().get_xticklabels():
        if ticklabel.get_text() in just_second:
            ticklabel.set_color('r')
    for ticklabel in plt.gca().get_yticklabels():
        if ticklabel.get_text() in just_second:
            ticklabel.set_color('r')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_preds_first', type=str, required=True)
    parser.add_argument('--input_preds_second', type=str, required=True)
    parser.add_argument('--id2pos', type=str, required=True)
    parser.add_argument('--database_first', type=str, required=True)
    args = parser.parse_args()
    main(args)
