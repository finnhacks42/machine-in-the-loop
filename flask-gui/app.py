# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, flash, Markup, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from sqlalchemy import desc

# Other libs
import ast
import os
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing  # to normalise existing X
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
from tqdm.auto import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.spatial import KDTree
from sentence_transformers import SentenceTransformer, util
import torch
import json
from numpyencode import EncodeFromNumpy, DecodeToNumpy
import time
from sklearn.manifold import TSNE
import seaborn as sns
import torch.nn.functional as F
from collections import Counter
from nltk.corpus import stopwords
import hdbscan


app = Flask(__name__)
app.secret_key = 'dev'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///covid.db'

# set default button sytle and size, will be overwritten by macro parameters
app.config['BOOTSTRAP_BTN_STYLE'] = 'primary'
app.config['BOOTSTRAP_BTN_SIZE'] = 'sm'
# app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'lumen'  # uncomment this line to test bootswatch theme

# set default icon title of table actions
app.config['BOOTSTRAP_TABLE_VIEW_TITLE'] = 'Read'
app.config['BOOTSTRAP_TABLE_EDIT_TITLE'] = 'Update'
app.config['BOOTSTRAP_TABLE_DELETE_TITLE'] = 'Remove'
app.config['BOOTSTRAP_TABLE_NEW_TITLE'] = 'Create'

app.config['UPLOAD_FOLDER'] = 'static/covid'
# 500 MB
app.config['MAX_CONTENT_PATH'] = 500000000

bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
csrf = CSRFProtect(app)

from forms import *
from models import *

# Define some global variables (current solution)
tweet_embed = None
teet_text = None
#tweet_stances = None
theme_centroids = {}
bad_theme_centroids = {}
K = 1
embedder = None

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    pass

@app.route('/', methods=['GET', 'POST'])
def index():
    form = FileUploadForm()
    print(request.form)
    global K
    if form.validate_on_submit() and 'restart' in request.form:
        # Restart the DB from scratch
        db.drop_all()
        db.create_all()

        theme_centroids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'files', 'theme_centroids.json')
        bad_theme_centroids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'files', 'bad_theme_centroids.json')

        if os.path.exists(theme_centroids_path):
            os.remove(theme_centroids_path)
        if os.path.exists(bad_theme_centroids_path):
            os.remove(bad_theme_centroids_path)

        K = form.k.data

        if form.method.data == 'kmeans':
            return redirect(url_for('loading'))
        elif form.method.data == 'hdbscan':
            return redirect(url_for('loading_hdbscan'))

    elif form.validate_on_submit() and 'submit' in request.form:
        assign_fixed_clusters()

        K = form.k.data

        if form.method.data == 'kmeans':
            return redirect(url_for('loading'))
        elif form.method.data == 'hdbscan':
            return redirect(url_for('loadinghdbscan'))

    return render_template('index.html', form=form)


@app.route('/loading', methods=['GET', 'POST'])
def loading():
    return render_template("loading.html")

@app.route('/loadinghdbscan', methods=['GET', 'POST'])
def loadinghdbscan():
    return render_template('loadinghdbscan.html')

@app.route('/loading_wordclouds', methods=['GET', 'POST'])
def loading_wordclouds():
    return render_template("loading_wordclouds.html")

def assign_fixed_clusters():
    themes = Theme.query.order_by('name').all()
    themes = [t for t in themes if t.name != "Unknown"]
    print([t.name for t in themes])
    # If there is nothing in DB, then just return
    if len(themes) <= 0:
        return

    query_vectors = []; bad_query_vectors = []

    print("GOOD", theme_centroids.keys())
    print("BAD", bad_theme_centroids.keys())


    tweet_embed_torch = torch.from_numpy(tweet_embed).float()

    pbar = tqdm(total=tweet_embed_torch.shape[0], desc='assigning fixed clusters')
    for i in range(0, tweet_embed_torch.shape[0]):
        tweet = Tweet.query.filter_by(tweet_id=str(i)).first()
        # if it is marked as good, we do not want to move it
        if tweet.good:
            continue

        cos_scores = []; bad_cos_scores = []
        for j in range(0, len(themes)):
            tweet_torch = tweet_embed_torch[i]
            query_vectors   = torch.from_numpy(theme_centroids[themes[j].name]).float()

            if themes[j].name in bad_theme_centroids:
                bad_query_vectors = torch.from_numpy(bad_theme_centroids[themes[j].name]).float()
            else:
                bad_query_vectors = torch.from_numpy(np.zeros(theme_centroids[themes[j].name].shape)).float()

            #print(tweet_torch.shape, query_vectors.shape, bad_query_vectors.shape)

            _cos_scores = util.cos_sim(tweet_torch, query_vectors)[0]
            _bad_cos_scores = util.cos_sim(tweet_torch, bad_query_vectors)[0]

            cos_score = torch.max(_cos_scores).item()
            bad_cos_score = torch.max(_bad_cos_scores).item()

            cos_scores.append(cos_score)
            bad_cos_scores.append(bad_cos_score)

        cos_scores = torch.FloatTensor(cos_scores)
        bad_cos_scores = torch.FloatTensor(bad_cos_scores)

        top_results = torch.topk(cos_scores, k=len(themes))

        top_score = None; top_index = None; index = 0
        while top_score is None:
            curr_index = top_results.indices[index]
            if bad_cos_scores[curr_index] > top_results.values[index]:
                index += 1
            else:
                top_score = top_results.values[index]
                top_index = top_results.indices[index]
        # If this happens in all cases (could it?), then just assign the first one
        if top_score is None:
            top_score = top_results.values[0]
            top_index = top_results.indixes[0]

        top_score = top_score.item()
        top_index = top_index.item()

        #top_index = [x.item() for x in top_results.indices][0]
        #top_score = [x.item() for x in top_results.values][0]
        #print(themes[top_index].name, cos_scores)

        tweet = Tweet.query.filter_by(tweet_id=str(i)).first()
        tweet.theme = themes[top_index]
        tweet.distance = round(1 - top_score, 4)

        # This was here to debug if there were any new assignments
        #if not themes[top_index].name.startswith('KMeans'):
        #    print(i, themes[top_index].name)

        pbar.update(1)
    pbar.close()
    db.session.commit()


@app.route('/hdbscan', methods=['POST', 'GET'])
def run_hdbscan():
    global theme_centroids

    themes = [t for t in Theme.query.order_by('name').all() if t.name != 'Unknown' and not t.name.startswith('HDBSCAN')]
    hdbscan_names = [t for t in Theme.query.order_by('name').all() if t.name.startswith('HDBSCAN') or t.name == 'Unknown']

    if len(themes) > 0 and len(hdbscan_names) > 0:
        tweet_ids = []
        for theme in hdbscan_names:
            for tw in theme.tweets:
                tweet_ids.append(int(tw.tweet_id))
        tweet_ids.sort()
    elif len(themes) > 0 and len(hdbscan_names) <= 0:
        # there is nothing else to be done
        return redirect(url_for('data'))
    else:
        # This is the first time we run it, run on everything
        tweet_ids = list(range(0, tweet_embed.shape[0]))

    #tweet_ids = tweet_ids[:5000]

    X_Norm = preprocessing.normalize(tweet_embed[tweet_ids])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=True)
    print(clusterer)
    print('fitting...')
    clusterer.fit(X_Norm)
    print('DONE.')

    #print(clusterer.labels_)

    for cluster_idx in range(0, clusterer.labels_.max() + 1):
        theme = "HDBSCAN_{}".format(cluster_idx)
        exists = db.session.query(Theme.id).filter_by(name=theme).first() is not None
        if not exists:
            thm = Theme(name=theme)
            db.session.add(thm)
    db.session.commit()

    # We are gonna use 1 - probability here instead of distance to centroid
    pbar = tqdm(total=len(tweet_ids), desc='calculating distances and storing tweets')
    for j, i in zip(tweet_ids, range(0, len(tweet_ids))):
        cluster_idx = clusterer.labels_[i]
        if cluster_idx >= 0:
            theme_name = "HDBSCAN_{}".format(cluster_idx)
            #print(theme_name)
            theme = Theme.query.filter_by(name=theme_name).first()
            #print(theme)

            distance_to_centroid = 1 - clusterer.probabilities_[i]
            distance_to_centroid = round(distance_to_centroid,4)
            tw = Tweet.query.filter_by(tweet_id=j).first()

            # Update tweet (We assume that the DB has been created before)
            tw.theme=theme
            tw.good=False
            tw.distance=distance_to_centroid
            #print(tw.theme.name)

        pbar.update(1)

    pbar.close()

    db.session.commit()
    return redirect(url_for('data'))


@app.route('/kmeans', methods=['POST', 'GET'])
def kmeans():
    global theme_centroids
    # TO-DO: write a copy instruction to maintain the database checkpoint

        # this is here to debug
    #return redirect(url_for('data'))

    themes = [t for t in Theme.query.order_by('name').all() if t.name != 'Unknown' and not t.name.startswith('KMeans')]
    kmeans_names = [t for t in Theme.query.order_by('name').all() if t.name.startswith('KMeans') or t.name == 'Unknown']

    if len(themes) > 0 and len(kmeans_names) > 0:
        tweet_ids = []
        for theme in kmeans_names:
            for tw in theme.tweets:
                tweet_ids.append(int(tw.tweet_id))
        tweet_ids.sort()
    elif len(themes) > 0 and len(kmeans_names) <= 0:
        # there is nothing else to be done
        return redirect(url_for('data'))
    else:
        # This is the first time we run it, run on everything
        tweet_ids = list(range(0, tweet_embed.shape[0]))

    print(tweet_ids)

    X_Norm = preprocessing.normalize(tweet_embed[tweet_ids])

    kmeans = KMeans(n_clusters=K, random_state=0, verbose=True).fit(X_Norm)
    # Create clusters if they don't exist
    for cluster_idx in range(0, K):
        theme = "KMeans_{}".format(cluster_idx)
        exists = db.session.query(Theme.id).filter_by(name=theme).first() is not None
        if not exists:
            thm = Theme(name=theme)
            db.session.add(thm)
    db.session.commit()

    centroids = kmeans.cluster_centers_
    # Update theme centroids in memory
    for cluster_idx in range(0, len(centroids)):
        theme = "KMeans_{}".format(cluster_idx)
        theme_centroids[theme] = centroids[cluster_idx]
        bad_theme_centroids[theme] = np.zeros(theme_centroids[theme].shape)

    pbar = tqdm(total=len(tweet_ids), desc='calculating distances and storing tweets')
    for j, i in zip(tweet_ids, range(0, len(tweet_ids))):
        cluster_idx = kmeans.labels_[i]
        theme_name = "KMeans_{}".format(cluster_idx)
        theme = Theme.query.filter_by(name=theme_name).first()

        distance_to_centroid = distance.cosine(tweet_embed[j], centroids[cluster_idx])
        distance_to_centroid = round(distance_to_centroid,4)
        tw = Tweet.query.filter_by(tweet_id=j).first()

        # Update tweet if DB has been created before
        if tw is not None:
            tw.theme=theme
            tw.good=False
            tw.distance=distance_to_centroid
        else:
           # Create the tweet
            tw = Tweet(tweet_id=j, text=tweet_text[j], theme=theme, distance=distance_to_centroid,
                       stance='neutral', mf='none', morality='non-moral')
            db.session.add(tw)

        pbar.update(1)

    pbar.close()

    update_theme_centroids_file_in_disk()
    update_bad_theme_centroids_file_in_disk()

    db.session.commit()
    return redirect(url_for('data'))


#@app.route('/worclouds', methods=['POST', 'GET'])
def wordclouds(theme_name):
    # clear directory
    wordcloud_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', 'wordcloud*')
    os.system('rm {}'.format(wordcloud_path))

    theme_docs = []
    theme_names = []

    themes = Theme.query.order_by('name').all()
    K_curr = 100; found_one = False
    pbar = tqdm(total=len(themes), desc='calculating wordcloud')
    for theme in themes:
        tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=theme.name).order_by('distance').limit(K_curr).all()
        if len(tweets) > 0 and theme.name != 'Unknown':
            all_text = " ".join([t.text for t in tweets])
            theme_names.append(theme.name)
            theme_docs.append(all_text)
            found_one = True
            #print(all_text)
            #print(theme)
            #print('-------')
        pbar.update(1)
    pbar.close()

    if found_one:
        vectorizer = TfidfVectorizer(max_df=0.7, ngram_range = (2,3))
        X = vectorizer.fit_transform(theme_docs)
        feature_names = vectorizer.get_feature_names()

        dense = X.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        viz_data = df.transpose()
        viz_data.columns = theme_names

    theme = Theme.query.filter_by(name=theme_name).first()

    ts = time.time()
    word_cloud_img = 'wordcloud_{}_{}.png'.format(theme.name, ts)

    if theme.name in viz_data:
        wordcloud = WordCloud(background_color='white').generate_from_frequencies(viz_data[theme.name])
        plt.figure(figsize=(5,5))
        plt.imshow(wordcloud)
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'images', word_cloud_img))
        #plt.show()
        plt.close()

    return word_cloud_img

def stance_distribution(theme, tweets, ts):
    stance_img = 'stance_{}_{}.png'.format(theme.name, ts)

    if len(tweets) > 0:
        stances = [t.stance for t in tweets]
        counts = [stances.count('pro-vax'), stances.count('anti-vax')]
        print(theme.name, counts)
        clrs = ['royalblue', 'crimson']
        f, ax = plt.subplots(figsize=(5,5)) # set the size that you'd like (width, height)
        plt.xticks(rotation=60)
        plt.bar(['pro-vax', 'anti-vax'], counts, color=clrs)
        #plt.bar(theme_names, counts, color=clrs, width=0.4)
        #ax.legend(fontsize = 10)
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'images', stance_img), bbox_inches = "tight")
        #plt.show()
        plt.close()
    return stance_img

def morality_distribution(theme, tweets, ts):
    moral_img = 'moral_{}_{}.png'.format(theme.name, ts)

    if len(tweets) > 0:
        morals = [t.morality for t in tweets]
        counts = [morals.count('moral'), morals.count('non-moral')]
        print(theme.name, counts)
        clrs = ['royalblue', 'crimson']
        f, ax = plt.subplots(figsize=(5,5)) # set the size that you'd like (width, height)
        plt.xticks(rotation=60)
        plt.bar(['moral', 'non-moral'], counts, color=clrs)
        #plt.bar(theme_names, counts, color=clrs, width=0.4)
        #ax.legend(fontsize = 10)
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'images', moral_img), bbox_inches = "tight")
        #plt.show()
        plt.close()
    return moral_img

def mf_distribution(theme, tweets, ts):
    mf_img = 'mf_{}_{}.png'.format(theme.name, ts)

    if len(tweets) > 0:
        mfs = [t.mf for t in tweets]
        mfs = ['none' if t.mf is None else t.mf for t in tweets]
        keys = list(set(mfs))
        counts = [mfs.count(k) for k in keys]
        f, ax = plt.subplots(figsize=(5,5))
        plt.xticks(rotation=60)
        my_cmap = plt.get_cmap("viridis")
        plt.bar(keys, counts, color=my_cmap(counts))
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'images', mf_img), bbox_inches = "tight")
        #plt.show()
        plt.close()
    return mf_img

def symbol_distribution(theme_name):
    os.system('rm {}/stance*'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images')))
    os.system('rm {}/moral*'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images')))
    os.system('rm {}/mf*'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images')))

    theme = Theme.query.filter_by(name=theme_name).first()
    tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=theme.name).order_by('distance').all()

    ts = time.time()
    stance_img = os.path.join('covid', 'images', stance_distribution(theme, tweets, ts))
    moral_img =  os.path.join('covid', 'images', morality_distribution(theme, tweets, ts))
    mf_img =  os.path.join('covid', 'images', mf_distribution(theme, tweets, ts))

    return stance_img, moral_img, mf_img

def update_theme_centroids(theme_name):
    global theme_centroids
    # Add the good phrases
    phrases = Phrase.query.filter_by(goodness='good').join(Phrase.theme, aliased=True).filter_by(name=theme_name).all()
    phrases = [p.text for p in phrases]
    if len(phrases) > 0:
        centroids = embedder.encode(phrases)
        theme_centroids[theme_name] = np.array(centroids)
    else:
        tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=theme_name).all()
        tweets = [t.tweet_id for t in tweets]
        centroids = tweet_embed[tweets]
        centroids = np.array(centroids)
        theme_centroids[theme_name] = np.mean(centroids, axis=0)
    update_theme_centroids_file_in_disk()

def update_bad_theme_centroids(theme_name):
    global bad_theme_centroids
    # Add the bad phrases
    phrases = Phrase.query.filter_by(goodness='good').join(Phrase.theme, aliased=True).filter_by(name=theme_name).all()
    phrases = [p.text for p in phrases]
    if len(phrases) > 0:
        centroids = embedder.encode(phrases)
        bad_theme_centroids[theme_name] = np.array(centroids)
        #np.mean(centroids, axis=0)
    update_bad_theme_centroids_file_in_disk()

def update_all_centroids():
    themes = Theme.query.all()
    pbar = tqdm(total=len(themes), desc='Updating centroids')
    for theme in themes:
        update_theme_centroids(theme.name)
        update_bad_theme_centroids(theme.name)

def update_theme_distances(theme_name):
    theme = Theme.query.filter_by(name=theme_name).first()
    tweets = theme.tweets
    for tw in tweets:
        vec = tweet_embed[tw.tweet_id]
        #print(mat.shape, vec.shape)
        distance_to_centroid = np.min(np.array([distance.cosine(vec,x) for x in theme_centroids[theme_name]]))
        tw.distance = round(distance_to_centroid,4)
    db.session.commit()

def mark_as_good():
    for checkbox in request.form.getlist('checkbox'):
        tweet = Tweet.query.get(checkbox)
        tweet.good = True
        phrase = Phrase.query.filter_by(text=tweet.text).first()
        if not phrase:
            new_phrase = Phrase(text=tweet.text, theme=tweet.theme, goodness='good', mf=tweet.mf, stance=tweet.stance)
            db.session.add(new_phrase)
        else:
            phrase.goodness = 'good'
    db.session.commit()

def mark_as_bad():
    for checkbox in request.form.getlist('checkbox'):
        tweet = Tweet.query.get(checkbox)
        tweet.good = False
        phrase = Phrase.query.filter_by(text=tweet.text).first()
        if not phrase:
            new_phrase = Phrase(text=tweet.text, theme=tweet.theme, goodness='bad', mf=tweet.mf, stance=tweet.stance)
            db.session.add(new_phrase)
        else:
            phrase.goodness = 'bad'
    db.session.commit()

def assign_to_theme(tweet_ids, theme_name):
    for tweet_id in tweet_ids:
        tweet = Tweet.query.get(tweet_id)
        theme = Theme.query.filter_by(name=theme_name).first()
        tweet.theme = theme
        tweet.good = True

        phrase = Phrase.query.filter_by(text=tweet.text).first()
        if not phrase:
            new_phrase = Phrase(text=tweet.text, theme=tweet.theme, goodness='good', mf=tweet.mf, stance=tweet.stance)
            db.session.add(new_phrase)
        else:
            phrase.goodness = 'good'

    db.session.commit()

@app.route('/data', methods=['GET', 'POST'])
def data():
    theme_choices = [t.name for t in Theme.query.order_by('name').all()]

    form_explore = DataForm(k=10)
    form_explore.theme.choices = theme_choices
    empty_table = True; tweets = None; columns=None

    form_assign = TweetEditForm(theme="N/A")
    form_assign.theme.choices = ["N/A"] + theme_choices

    form_search = TextQueryForm(k=10)

    if form_explore.validate_on_submit():
        if 'close' in request.form:
            tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=form_explore.theme.data).order_by('distance').limit(100).all()
        else:
            tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=form_explore.theme.data).order_by(desc(Tweet.distance)).limit(100).all()

        if len(tweets) > 0:
            empty_table = False
            first = Tweet.query.first()
            columns = first.__table__.columns.keys()

    if form_search.validate_on_submit():
        tweet_embed_torch = torch.from_numpy(tweet_embed)
        query_embedding = embedder.encode(form_search.query.data)

        cos_scores = util.cos_sim(query_embedding, tweet_embed_torch)[0]
        top_results = torch.topk(cos_scores, k=form_search.k.data)

        top_scores = [x.item() for x in top_results.values]
        top_results = [str(x.item()) for x in top_results.indices]

        print(top_scores)
        print(top_results)

        tweets = [Tweet.query.filter_by(tweet_id=id).one() for id in top_results]

        if len(tweets) > 0:
            empty_table = False
            first = Tweet.query.first()
            columns = first.__table__.columns.keys()

    print(columns)

    if request.method == 'POST':
        print(request.form)
        if 'mark-as-good-button' in request.form and request.form['mark-as-good-button'] == "Clicked":
            mark_as_good()
            return redirect(url_for('data'))
        if 'mark-as-bad-button' in request.form and request.form['mark-as-bad-button'] == "Clicked":
            mark_as_bad()
            return redirect(url_for('data'))
        if 'tweetids' in request.form:
            # Assign to theme scenario
            theme_choice = request.form['theme']
            theme_name = request.form['name']

            if theme_name == "" and theme_choice == "N/A":
                flash("You have to specify either an existing theme or a new name")
            else:
                if theme_name == "":
                    theme_name = theme_choice
                else:
                    # Adding new code
                    add_theme(theme_name)

                # continuing with the re-assignment
                tweet_ids = request.form['tweetids'].strip().split()
                assign_to_theme(tweet_ids, theme_name)
                update_theme_centroids(theme_name)
                update_theme_distances(theme_name)
                return redirect(url_for('data'))

        if 'explore-similar-button' in request.form and request.form['explore-similar-button'] == "Clicked":
            return redirect(url_for('similar', tweet_ids=request.form.getlist('checkbox')))

    return render_template("data.html", rows=tweets, columns=columns, form=form_explore,
                            not_empty=not empty_table, form_assign=form_assign, form_search=form_search)


def update_theme_centroids_file_in_disk():
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'files', 'theme_centroids.json'), 'w') as fp:
        json.dump(theme_centroids, fp, cls=EncodeFromNumpy)

def update_bad_theme_centroids_file_in_disk(update_bad=False):
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'files', 'bad_theme_centroids.json'), 'w') as fp:
        json.dump(bad_theme_centroids, fp, cls=EncodeFromNumpy)

def delete_theme(theme_name):
    global theme_centroids
    global bad_theme_centroids

    theme = Theme.query.filter_by(name=theme_name).first()
    unknown = Theme.query.filter_by(name="Unknown").first()
    # Clear all associations
    for tweet in theme.tweets:
        tweet.theme = unknown
    db.session.commit()
    # Remove from DB
    db.session.delete(theme)
    db.session.commit()

    del theme_centroids[theme_name]
    del bad_theme_centroids[theme_name]

    update_theme_centroids_file_in_disk()
    update_bad_theme_centroids_file_in_disk()

    # Remove wordcloud file
    os.remove('{}/wordcloud_{}.png'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), theme_name))
    os.remove('{}/stance_{}.png'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), theme_name))
    flash("Deleted {}".format(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), theme_name))

def edit_theme(theme_name, new_name):
    theme = Theme.query.filter_by(name=theme_name).first()
    theme.name = new_name

    # Switch centroids
    global theme_centroids
    global bad_theme_centroids
    theme_centroids[new_name] = theme_centroids[theme_name]
    bad_theme_centroids[new_name] = bad_theme_centroids[theme_name]

    # remove old one
    del theme_centroids[theme_name]
    del bad_theme_centroids[theme_name]
    update_theme_centroids_file_in_disk()
    update_bad_theme_centroids_file_in_disk()

    # Rename file
    os.rename('{}/wordcloud_{}.png'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), theme_name),
              '{}/wordcloud_{}.png'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), new_name))

    os.rename('{}/stance_{}.png'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), theme_name),
              '{}/stance_{}.png'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), new_name))

    db.session.commit()
    flash("{} -> {}".format(theme_name, new_name))

def add_theme(theme_name):
    global theme_centroids
    global bad_theme_centroids

    theme = Theme(name=theme_name)
    db.session.add(theme)
    db.session.commit()
    theme_centroids[theme_name] = np.zeros((768,))
    bad_theme_centroids[theme_name] = np.zeros((768,))

    flash("Added {}".format(theme_name))

@app.route('/themes', methods=['GET', 'POST'])
def themes():
    form_explore = ThemeForm()
    form_explore.theme.choices = [t.name for t in Theme.query.order_by('name').all()]

    form_new_code = NewCodeForm()
    form_new_phrase = NewPhraseForm()

    theme = None;
    word_cloud_img = None; stance_img=None; moral_img=None; mf_img = None
    exists_file = False
    good_phrases = None; bad_phrases = None
    pos_entities = None; neg_entities = None

    print(request.form)
    if request.method == 'POST':
        if form_explore.validate_on_submit() and 'visualize' in request.form:
            theme_name = form_explore.theme.data

            word_cloud_img = wordclouds(theme_name)
            stance_img, moral_img, mf_img = symbol_distribution(theme_name)
            exists_file = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'images', word_cloud_img))
            word_cloud_img = os.path.join('covid', 'images', word_cloud_img)

            theme = Theme.query.filter_by(name=theme_name).first()
            good_phrases = Phrase.query.filter_by(goodness='good').join(Phrase.theme, aliased=True).filter_by(name=theme_name).all()
            bad_phrases = Phrase.query.filter_by(goodness='bad').join(Phrase.theme, aliased=True).filter_by(name=theme_name).all()

            pos_entities = Tweet_Has_Entity.query.filter_by(sentiment="actor-pos")\
                            .join(Tweet_Has_Entity.tweet, aliased=True).join(Tweet.theme)\
                            .filter_by(name=theme_name).all()
            pos_entities = [e.entity_text for e in pos_entities if e.entity_text not in stopwords.words('english')]

            neg_entities = Tweet_Has_Entity.query.filter_by(sentiment="actor-neg")\
                            .join(Tweet_Has_Entity.tweet, aliased=True).join(Tweet.theme)\
                            .filter_by(name=theme_name).all()
            neg_entities = [e.entity_text for e in neg_entities if e.entity_text not in stopwords.words('english')]

            pos_entities = Counter(pos_entities).most_common(10)
            neg_entities = Counter(neg_entities).most_common(10)

        if form_explore.validate_on_submit() and 'delete' in request.form:
            theme_name = form_explore.theme.data
            delete_theme(theme_name)
            return redirect(url_for('themes'))
        if form_new_code.validate_on_submit() and 'new_code' in request.form:
            add_theme(form_new_code.name.data)
            return redirect(url_for('themes'))
        if form_new_code.validate_on_submit() and 'edit_code' in request.form and 'theme_name_f1' in request.form:
            theme_name = request.form['theme_name_f1']
            edit_theme(theme_name, form_new_code.name.data)
            return redirect(url_for('themes'))


    if form_new_phrase.validate_on_submit() and 'theme_name_f2' in request.form:
        theme_name = request.form['theme_name_f2']
        theme = Theme.query.filter_by(name=theme_name).first()
        phrase = Phrase(text=form_new_phrase.text.data, theme=theme,
                        goodness=form_new_phrase.goodness.data,
                        mf=form_new_phrase.mf.data, stance=form_new_phrase.stance.data)
        update_theme_centroids(theme_name)
        update_bad_theme_centroids(theme_name)
        db.session.add(phrase)
        db.session.commit()
        return redirect(url_for('themes'))

    return render_template("themes.html", form_explore=form_explore, theme=theme, word_cloud_img=word_cloud_img,
                           stance_img=stance_img, moral_img=moral_img, mf_img=mf_img,
                           form_new_code=form_new_code, exists_file=exists_file,
                           form_new_phrase=form_new_phrase, good_phrases=good_phrases, bad_phrases=bad_phrases,
                           pos_entities=pos_entities, neg_entities=neg_entities)

def plot_tsne(ts):
    target = [] #Index of the cluster each sample/tweet belongs to.
    X = [] # {array-like, sparse matrix} of shape (n_samples, n_features), for our case tweet embedding

    themes = Theme.query.all()
    pbar = tqdm(total=len(themes), desc='preparing tsne data')
    for i, theme in enumerate(themes):
        if theme.name == "Unknown":
            continue
        # Add the centroid first
        #centroid = theme_centroids[theme.name]
        #X.append(centroid)
        #target.append(theme.name)

        # Plot things that are near
        tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=theme.name).order_by('distance').limit(500).all()
        #print(theme.name, [tw.distance for tw in tweets])
        ids = [int(tw.tweet_id) for tw in tweets]

        # Plot things that are far
        #tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=theme.name).order_by(desc(Tweet.distance)).limit(100).all()
        #ids += [int(tw.tweet_id) for tw in tweets]

        vectors = tweet_embed[ids]
        X += list(vectors)
        target += [theme.name] * len(ids)
        pbar.update(1)

    pbar.close()

    tSNE=TSNE(n_components=2, perplexity=40, n_iter=300)
    tSNE_result=tSNE.fit_transform(np.array(X))

    x=tSNE_result[:,0]
    y=tSNE_result[:,1]  

    df = pd.DataFrame(X)
    df['themes']=target
    df['x']=x
    df['y']=y

    plt.figure(figsize=(16,7))
    sns.scatterplot(x='x',y='y',hue='themes',palette=sns.color_palette("hls",df.themes.nunique()),data=df,
               legend="full")

    tsne_img = 'tsne_{}.png'.format(ts)
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'images', tsne_img), bbox_inches = "tight")
    plt.close()

    return tsne_img


@app.route('/coverage', methods=['GET', 'POST'])
def coverage():
    os.system('rm {}/cluster_assignment*'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images')))
    os.system('rm {}/named_coverage*'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images')))
    os.system('rm {}/tsne*'.format(os.path.join(app.config['UPLOAD_FOLDER'], 'images')))

    ts = time.time()
    # Plot assignments (bar graph)
    themes = Theme.query.all()
    theme_names = []; counts = []
    for theme in themes:
        num_tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=theme.name).count()
        if num_tweets > 0:
            counts.append(num_tweets)
            theme_names.append(theme.name)

    clrs = ['grey' if name.startswith('KMeans') or name == 'Unknown' else 'red' for name in theme_names]
    f, ax = plt.subplots(figsize=(15,5)) # set the size that you'd like (width, height)
    plt.xticks(rotation=60)
    plt.bar(theme_names, counts, color=clrs)
    #plt.bar(theme_names, counts, color=clrs, width=0.4)
    ax.legend(fontsize = 10)
    bar_plot_img = 'cluster_assignment_{}.png'.format(ts)
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'images', bar_plot_img), bbox_inches = "tight")
    #plt.show()
    plt.close()

    # Plot coverage indicators (bar or pie graph)
    not_named = 0; named = 0
    for name, count in zip(theme_names, counts):
        if name == 'Unknown' or name.startswith('KMeans_'):
            not_named += count
        else:
            named += count

    not_named_prop = (not_named) / (not_named + named)
    named_prop = (named) / (not_named + named)

    labels = ['Named', 'Not Named']
    sizes = [named_prop, not_named_prop]
    explode = (0.1, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
    ax1.axis('equal')
    pie_plot_img = 'named_coverage_{}.png'.format(ts)
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'images', pie_plot_img))
    #plt.show()
    plt.close()

    # Output silhoutte score
    # Plot clusters using t-sne (maybe? it takes a long time)
    tsne_img = plot_tsne(ts)

    # Fix the path
    bar_plot_img = os.path.join('covid', 'images', bar_plot_img)
    pie_plot_img = os.path.join('covid', 'images', pie_plot_img)
    tsne_img = os.path.join('covid', 'images', tsne_img)


    return render_template("coverage.html", bar_plot_img=bar_plot_img, pie_plot_img=pie_plot_img, tsne_img=tsne_img)

@app.route('/data/<tweet_ids>', methods=['GET', 'POST'])
def similar(tweet_ids):
    theme_choices = ["N/A"] + [t.name for t in Theme.query.order_by('name').all()]
    form_assign_theme = TweetEditForm(theme="N/A")
    form_assign_theme.theme.choices = theme_choices

    tweet_ids = ast.literal_eval(tweet_ids)
    query_tweets = Tweet.query.filter(Tweet.id.in_(tweet_ids)).all()
    tweet_ids = [t.tweet_id for t in query_tweets]

    average = torch.from_numpy(np.mean(tweet_embed[tweet_ids], axis=0).reshape(1, -1))
    tweet_embed_torch = torch.from_numpy(tweet_embed)

    print(average.shape)
    print(tweet_embed_torch.shape)

    cos_scores = util.cos_sim(average, tweet_embed_torch)[0]
    top_results = torch.topk(cos_scores, k=10)

    top_scores = [x.item() for x in top_results.values]
    top_results = [str(x.item()) for x in top_results.indices]

    print(top_scores)
    print(top_results)

    tweets = [Tweet.query.filter_by(tweet_id=id).one() for id in top_results]

    empty_table = False
    first = Tweet.query.first()
    columns = first.__table__.columns.keys()

    if request.method == 'POST':
        if 'mark-as-good-button' in request.form and request.form['mark-as-good-button'] == "Clicked":
            mark_as_good()
            return redirect(url_for('data'))
        if 'mark-as-bad-button' in request.form and request.form['mark-as-bad-button'] == "Clicked":
            mark_as_bad()
            return redirect(url_for('data'))
        if 'tweetids' in request.form:
            # Assign to theme scenario
            theme_choice = request.form['theme']
            theme_name = request.form['name']

            if theme_name == "" and theme_choice == "N/A":
                flash("You have to specify either an existing theme or a new name")
            else:
                if theme_name == "":
                    theme_name = theme_choice
                else:
                    # Adding new code
                    theme = Theme(name=theme_name)
                    db.session.add(theme)
                    db.session.commit()
                # continuing with the re-assignment
                tweet_ids = request.form['tweetids'].strip().split()
                assign_to_theme(tweet_ids, theme_name)
                update_theme_centroids(theme_name)
                update_theme_distances(theme_name)
                return redirect(url_for('data'))
        if 'explore-similar-button' in request.form and request.form['explore-similar-button'] == "Clicked":
            return redirect(url_for('similar', tweet_ids=request.form.getlist('checkbox')))

    return render_template('similar.html', rows=tweets, columns=columns, not_empty=not empty_table,
                           form=form_assign_theme, cos_scores=cos_scores, tweets=query_tweets)

@app.route('/delete_theme_tweet', methods=['GET', 'POST'])
def delete_theme_tweet():
    print('delete_theme_tweet')
    if request.method == 'POST':
        print(request.form)
    return redirect(url_for('index'))

@app.route('/delete_theme_phrase', methods=['GET', 'POST'])
def delete_theme_phrase():
    print('delete_theme_phrase')
    if request.method == 'POST':
        print(request.form)
    return redirect(url_for('index'))

@app.route('/delete_phrase/<phrase_id>/delete', methods=['GET', 'POST'])
def delete_phrase(phrase_id):
    phrase = Phrase.query.get(phrase_id)
    db.session.delete(phrase)
    db.session.commit()
    if phrase:
        return f'Deleting {phrase_id}. Return to <a href="/themes">themes</a>.'
    return f'Could not delete message {phrase_id} as it does not exist. Return to <a href="/themes">themes</a>'

@app.route('/edit_phrase/<phrase_id>/edit', methods=['GET', 'POST'])
def edit_phrase(phrase_id):
    phrase = Phrase.query.get(phrase_id)
    form_edit_phrase = EditPhraseForm()

    if form_edit_phrase.validate_on_submit():
        phrase.goodness=form_edit_phrase.goodness.data
        phrase.mf=form_edit_phrase.mf.data
        phrase.stance=form_edit_phrase.stance.data
        db.session.commit()
        return f'Edited {phrase_id}. Return to <a href="/themes">themes</a>.'

    return render_template("edit_phrase.html", form_edit_phrase=form_edit_phrase, phrase=phrase)

@app.before_first_request
def create_tables():
    #db.drop_all()
    db.create_all()

    global tweet_embed
    global tweet_text
    global embedder

    global theme_centroids
    global bad_theme_centroids

    tweet_embed_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'files', 'sbert.npy')
    tweet_text_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'files', 'text.npy')

    if os.path.exists(tweet_embed_dir):
        tweet_embed = np.load(tweet_embed_dir)
    if os.path.exists(tweet_text_dir):
        tweet_text = np.load(tweet_text_dir)

    # load theme centroid files from disk if they exist
    embedder = SentenceTransformer('all-mpnet-base-v2')
    update_all_centroids()

    # Add an unknown theme if it does not exist
    exists = db.session.query(Theme.id).filter_by(name='Unknown').first() is not None
    if not exists:
        thm = Theme(name='Unknown')
        db.session.add(thm)
        db.session.commit()

    # Remove themes that have zero assignments
    themes = Theme.query.all()
    theme_names = []; counts = []
    for theme in themes:
        num_tweets = Tweet.query.join(Tweet.theme, aliased=True).filter_by(name=theme.name).count()
        if num_tweets == 0:
            # delete phrases (if they were lingering)
            phrases = Phrase.query.filter_by(theme_id=theme.id).all()
            for phrase in phrases:
                db.session.delete(phrase)
            # Delete theme
            db.session.delete(theme)
            if theme.name in theme_centroids:
                del theme_centroids[theme.name]
            if theme.name in bad_theme_centroids:
                del bad_theme_centroids[theme.name]
    db.session.commit()


if __name__ == '__main__':
    app.run(debug=True)

