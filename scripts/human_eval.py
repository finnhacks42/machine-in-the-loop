import csv
import argparse
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance, masi_distance
from sklearn.metrics import *


def main(args):
    predictions = [];
    ann_1 = []; ann_2 = []; ann_3 = []
    annotation_triplets = []
    with open(args.annotations) as fp:
        spamreader = csv.reader(fp, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0:
                row = [r.strip() for r in row]
                theme_index = row.index('Theme')
                lb_index = row.index('Prediction')
                ml_index = row.index('Maria')
                if 'Dan' in row:
                    ch_index = row.index('Dan')
                else:
                    ch_index = row.index('Chelsy')
                tz_index = row.index('Tunaz')
            else:
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

                try:
                    ann_1.append(int(ml_label))
                    ann_2.append(int(ch_label))
                    ann_3.append(int(tz_label))
                    predictions.append(int(label))

                    annotation_triplets.append(('coder_1', 'sample_{}'.format(i), ml_label))
                    annotation_triplets.append(('coder_2', 'sample_{}'.format(i), ch_label))
                    annotation_triplets.append(('coder_3', 'sample_{}'.format(i), tz_label))
                except:
                    print(i, row)
                    continue

    print("Evaluated", len(ann_1), len(predictions), "examples")

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
    args = parser.parse_args()
    main(args)
