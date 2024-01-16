import csv
import argparse
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance, masi_distance
from sklearn.metrics import *
import numpy as np
import sqlite3

def main(args):
    predictions = [];
    ann_1 = []; ann_2 = []; ann_3 = []
    annotation_triplets = []
    distances_from_centroid = []
    with open(args.annotations) as fp:
        spamreader = csv.reader(fp, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0:
                theme_index = len(row) - 5
                lb_index = len(row) - 4
                ml_index = len(row) - 1
                ch_index = len(row) - 2
                tz_index = len(row) - 3
            elif args.theme is not None and row[theme_index] != args.theme:
                continue
            elif int(row[lb_index]) == 1:
                theme = row[theme_index]
                label = row[lb_index]
                ml_label = row[ml_index]
                ch_label = row[ch_index]
                tz_label = row[tz_index]

                print(theme)

                predictions.append(int(label))
                ann_1.append(int(ml_label))
                ann_2.append(int(ch_label))
                try:
                    ann_3.append(int(tz_label))
                except:
                    ann_3.append(int(ch_label))

                annotation_triplets.append(('coder_1', 'sample_{}'.format(i), ml_label))
                annotation_triplets.append(('coder_2', 'sample_{}'.format(i), ch_label))
                annotation_triplets.append(('coder_3', 'sample_{}'.format(i), tz_label))

    # Get majority labels
    ann_res = []
    for a1, a2, a3 in zip(ann_1, ann_2, ann_3):
        if sum([a1, a2, a3]) >= 2:
            ann_res.append(1)
        elif sum([a1, a2, a3]) <= 1:
            ann_res.append(0)

    print(predictions)
    print(ann_res)

    print(classification_report(predictions, ann_res, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--theme', type=str, default=None)
    args = parser.parse_args()
    main(args)
