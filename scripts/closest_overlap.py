import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_theme_group1(theme):
    if theme == "VaxSymptoms":
        return "VaxSideEffects"
    elif theme == "VaxAppointmentInfo":
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

def normalize_theme_group2(theme):
    if theme == "ReasonsForUSLaggingOnVaccines":
        return "ReasonsUSLagsOnVax"
    elif theme == "VaxDistributionIssueDueToLocalPolicy":
        return "VaxDistributionIssues"
    elif theme == "VaxAvailabilityInfo":
        return "VaxAvailability"
    elif theme == "#IGotMyVaccine":
        return "#IGotMyVax"
    else:
        return theme

def main(args):
    con1 = sqlite3.connect(args.db1)
    cur1 = con1.cursor()

    con2 = sqlite3.connect(args.db2)
    cur2 = con2.cursor()

    # Find themes first
    query = 'SELECT id, name FROM theme'

    cur1.execute(query)
    cur2.execute(query)

    themes1 = cur1.fetchall()
    themes2 = cur2.fetchall()

    # Only keep named themes
    themes1 = [(id, name) for (id, name) in themes1 if name != 'Unknown' and not name.startswith('KMeans')]
    themes2 = [(id, name) for (id, name) in themes2 if name != 'Unknown' and not name.startswith('KMeans')]


    tweets1 = {}; tweets2 = {}
    # For each theme, find the set of tweets that are assigned to it
    for (id, name) in themes1:
        query = 'SELECT DISTINCT tweet_id FROM tweet WHERE theme_id = {}'.format(id)
        cur1.execute(query)
        tweets = [x[0] for x in cur1.fetchall()]
        tweets1[name] = set(tweets)

    for (id, name) in themes2:
        query = 'SELECT DISTINCT tweet_id FROM tweet WHERE theme_id = {}'.format(id)
        cur2.execute(query)
        tweets = [x[0] for x in cur2.fetchall()]
        tweets2[name] = set(tweets)

    # Close databases
    cur1.close()
    cur2.close()
    con1.close()
    con2.close()

    overlaps = {}
    # Now find the biggest overlap coefficient
    for t1 in tweets1:
        overlaps[t1] = {}
        for t2 in tweets2:
            intersect = tweets1[t1] & tweets2[t2]
            min_len = min(len(tweets1[t1]), len(tweets2[t2]))
            overlaps[t1][t2] = len(intersect) / min_len

    overlap_matrix = np.zeros((len(themes1), len(themes2)))
    y_axis_names = []; x_axis_names = []

    for i, theme in enumerate(overlaps):
        print(theme)
        y_axis_names.append(normalize_theme_group1(theme))
        max_overlap = 0
        for j, theme2 in enumerate(overlaps[theme]):
            if i == 0:
                x_axis_names.append(normalize_theme_group2(theme2))
            overlap_matrix[i][j] = overlaps[theme][theme2]

            if overlaps[theme][theme2] > max_overlap:
                max_overlap = overlaps[theme][theme2]
                max_theme = theme2
        print('\t{}: {}'.format(max_theme, max_overlap))


    plt.figure(layout='constrained')
    ax = sns.heatmap(overlap_matrix, fmt='.2', annot=True, cmap='YlOrBr')
    ax.set_yticklabels(y_axis_names, rotation=0)
    ax.set_xticklabels(x_axis_names, rotation=90)
    ax.set_ylabel('First Group')
    ax.set_xlabel('Second Group')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db1', type=str, required=True)
    parser.add_argument('--db2', type=str, required=True)
    args = parser.parse_args()
    main(args)
