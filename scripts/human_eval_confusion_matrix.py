import csv
import argparse
from sklearn.metrics import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from aggregated_distance_metrics import get_all_quartiles, get_theme_quartiles
import sqlite3

def normalize_theme(theme):
    if theme == "VaxSymptoms":
        return "VaxSideEffects"
    elif theme == "VaxAppointmentInfo" or theme == "VaxAvailabilityInfo":
        return "VaxAppointments"
    elif theme == "IGotTheVax":
        return "GotTheVax"
    elif theme == "VaxApprovalInfo":
        return "VaxApproval"
    elif theme == "AntiVaxersSpreadMissinfo":
        return "AntiVaxSpreadMisinfo"
    elif theme == "ProVaxersSpreadMissinfo":
        return "ProVaxLie"
    elif theme == "AlternativeTreatmentsGood":
        return "AltTreatmentsGood"
    elif theme == "AlternativeTreatmentsBad":
        return "AltTreatmentsBad"
    elif theme == "EmphasizeFreeChoice":
        return "FreeChoiceVax"
    elif theme == "FreeChoiceAbortion":
        return "FreeChoiceOther"
    else:
        return theme

def main(args):
    conn = sqlite3.connect(args.database)
    cur = conn.cursor()
    (Q1, Q2, Q3, Q4) = get_all_quartiles(cur, isimmi=args.is_immi)
    # Keep only tweet_id
    Q1 = [t[0] for t in Q1]
    Q2 = [t[0] for t in Q2]
    Q3 = [t[0] for t in Q3]
    Q4 = [t[0] for t in Q4]

    text_mat = list(np.load(args.text))
    print(len(text_mat))
    #print(text_mat[0])
    #exit()

    ids = []; y_pred = []; y_gold = []; labels = set()
    with open(args.annotations) as fp:
        spamreader = csv.reader(fp, delimiter=',')
        for i, row in enumerate(spamreader):
            row = [r.strip() for r in row]
            if i == 0:
                tweet_index = row.index('Tweet')
                theme_index = row.index('Theme')
                appropriate_theme_index = row.index('Appropriate_Theme')
            else:
                tweet_id = text_mat.index(row[tweet_index])
                ids.append(tweet_id)

                theme = normalize_theme(row[theme_index])
                appropriate_theme = normalize_theme(row[appropriate_theme_index])
                y_pred.append(theme)
                if appropriate_theme == "":
                    y_gold.append(theme)
                else:
                    y_gold.append(appropriate_theme)
                labels.add(theme)
                if appropriate_theme != "Other" and appropriate_theme != "":
                    labels.add(appropriate_theme)

    num_q4 = 0; num_q3 = 0; num_q2 = 0; num_q1 = 0
    #print(Q1[:5])
    #exit()

    for i in range(0, len(y_gold)):
        if y_gold[i] == 'Other':
            twid = ids[i]
            if twid in Q4 and twid not in Q3:
                num_q4 += 1
            elif twid in Q3 and twid not in Q2:
                num_q3 += 1
            elif twid in Q2 and twid not in Q1:
                num_q2 += 1
            else:
                num_q1 += 1

    print("Q1, Q3, Q3 and Q4 for tweets that are Other")
    print(num_q1, num_q2, num_q3, num_q4)

    labels = np.array(list(labels) + ["Other"])
    cm = confusion_matrix(y_gold, y_pred, labels=labels, normalize='pred')
    order = np.argsort(-cm.diagonal())
    print(order)
    print(labels[order])

    plt.figure(layout='constrained')
    cm = cm[order,:][:,order]
    ax = sns.heatmap(cm, annot=True, fmt='.0%', cmap='Blues')
    ax.set_yticklabels(labels[order], rotation=0)
    ax.set_xticklabels(labels[order], rotation=90)
    ax.set_ylabel('True Category')
    ax.set_xlabel('Predicted Category')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, required=True)
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--is_immi', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
