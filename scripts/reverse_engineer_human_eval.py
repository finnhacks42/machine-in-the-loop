import csv
import argparse
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance, masi_distance
from sklearn.metrics import *
import numpy as np
import sqlite3

def main(args):
    curr_texts = list(np.load(args.text))

    conn = sqlite3.connect(args.database)
    cur = conn.cursor()

    # Find the ones that are positive in human eval and retrieve their distance and theme
    relevant_ds = {}; themes = set()
    with open(args.text_anns) as fp:
        spamreader = csv.reader(fp, delimiter=';')
        for i, row in enumerate(spamreader):
            if row[-1] == "1":
                text = row[0]
                tweet_id = curr_texts.index(text)
                #print(text)
                # Get theme and distance
                table_name = "tweet"
                if args.is_immi:
                    table_name = "tweet_immi"

                query = 'select theme_id, text, distance from {} where tweet_id = "{}"'
                query = query.format(table_name, tweet_id)
                (theme_id, _text, distance) = cur.execute(query).fetchone()

                query = 'select name from theme where id = "{}"'
                query = query.format(theme_id)
                (theme_name,) = cur.execute(query).fetchone()
                themes.add(theme_name)

                relevant_ds[i+1] = {'theme': row[-2], 'distance': distance}
                #print(text)
                #print(tweet_id)
                #exit()

    #print(relevant_ds)
    #exit()

    # Sort annotated elems by distance
    theme2id = {}; theme2distance = {}
    with open(args.annotations) as fp:
        spamreader = csv.reader(fp, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0 or i not in relevant_ds:
                continue
            theme_name = relevant_ds[i]['theme']
            distance = relevant_ds[i]['distance']

            if theme_name not in theme2id:
                theme2id[theme_name] = []
                theme2distance[theme_name] =    []
            theme2id[theme_name].append(i)
            theme2distance[theme_name].append(distance)

    id2quartile = {};
    for theme in theme2distance:
        print(theme, len(theme2id[theme]))
        '''
        if theme.startswith('KMeans'):
            for idx in theme2id[theme]:
                id2quartile[idx] = 'Q4'
        else:
        '''
        sorted_indices = np.argsort(theme2distance[theme])

        for i in range(0, 3):
            if i < len(sorted_indices):
                index = sorted_indices[i]
                id2quartile[theme2id[theme][index]] = 1
        for i in range(3, 6):
            if i < len(sorted_indices):
                index = sorted_indices[i]
                id2quartile[theme2id[theme][index]] = 2
        for i in range(6, 9):
            if i < len(sorted_indices):
                index = sorted_indices[i]
                id2quartile[theme2id[theme][index]] = 3
        for i in range(9, 12):
            if i < len(sorted_indices):
                index = sorted_indices[i]
                id2quartile[theme2id[theme][index]] = 4

    #print(id2quartile)
    #exit()

    predictions = [];
    ann_1 = []; ann_2 = []; ann_3 = []
    annotation_triplets = []
    distances_from_centroid = []
    with open(args.annotations) as fp:
        spamreader = csv.reader(fp, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0:
                row = [r.strip() for r in row]
                #print(row)
                theme_index = row.index('Theme')
                lb_index = row.index('Prediction')
                ml_index = row.index('Maria')
                if 'Dan' in row:
                    ch_index = row.index('Dan')
                else:
                    ch_index = row.index('Chelsy')
                tz_index = row.index('Tunaz')
            elif i not in relevant_ds:
                continue
            elif id2quartile[i] <= args.quartile or args.quartile is None and row[lb_index] == 1:
                row = [r.strip() for r in row]
                theme = row[theme_index]
                label = row[lb_index]
                ml_label = row[ml_index]
                if row[ch_index] == '':
                    ch_label = ml_label
                else:
                    ch_label = row[ch_index]

                if row[tz_index] == '':
                    tz_label = ml_label
                else:
                    tz_label = row[tz_index]

                print(row)
                print(ml_label, ch_label, tz_label)

                predictions.append(int(label))
                ann_1.append(int(ml_label))
                ann_2.append(int(ch_label))
                ann_3.append(int(tz_label))

                annotation_triplets.append(('coder_1', 'sample_{}'.format(i), ml_label))
                annotation_triplets.append(('coder_2', 'sample_{}'.format(i), ch_label))
                annotation_triplets.append(('coder_3', 'sample_{}'.format(i), tz_label))

                distances_from_centroid.append(relevant_ds[i]['distance'])


    print("Average Distance from Centroid")
    print(np.mean(distances_from_centroid))

    task = AnnotationTask(distance=binary_distance)
    task.load_array(annotation_triplets)
    print("Krippendorff's-Alpha (Binary Distance)", task.alpha())

    # Get majority labels
    ann_res = []
    for a1, a2, a3 in zip(ann_1, ann_2, ann_3):
        if sum([a1, a2, a3]) >= 2:
            ann_res.append(1)
        elif sum([a1, a2, a3]) <= 1:
            ann_res.append(0)

    print(classification_report(predictions, ann_res, digits=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--text_anns', type=str, required=True)
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--quartile', type=int, default=None)
    parser.add_argument('--is_immi', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
