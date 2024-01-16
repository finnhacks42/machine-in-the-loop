from __main__ import db
from sqlalchemy import ForeignKey
from sqlalchemy import JSON

class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.Integer, nullable=False, unique=True)
    text = db.Column(db.String(1000), nullable=False)
    distance = db.Column(db.Float)
    good = db.Column(db.Boolean, default=False, nullable=False)

    stance = db.Column(db.String, nullable=False)
    morality = db.Column(db.String, nullable=False)
    mf = db.Column(db.String, nullable=False)

    theme_id = db.Column(db.Integer, ForeignKey('theme.id'), nullable=False)
    theme = db.relationship('Theme', backref=db.backref('tweets', lazy=True))

class TweetImmi(db.Model):
    id =  db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.Integer, nullable=False, unique=True)
    text = db.Column(db.String(1000), nullable=False)
    distance = db.Column(db.Float)
    good = db.Column(db.Boolean, default=False, nullable=False)

    narrative = db.Column(db.String, nullable=False, default='None')
    immi_frame = db.Column(db.String, nullable=False, default='None')
    immi_role = db.Column(db.String, nullable=False, default='None')

    # Policy frames
    frame_political = db.Column(db.Boolean, default=False, nullable=False)
    frame_policy = db.Column(db.Boolean, default=False, nullable=False)
    frame_crime = db.Column(db.Boolean, default=False, nullable=False)
    frame_health = db.Column(db.Boolean, default=False, nullable=False)
    frame_security = db.Column(db.Boolean, default=False, nullable=False)
    frame_economic = db.Column(db.Boolean, default=False, nullable=False)

    theme_id = db.Column(db.Integer, ForeignKey('theme.id'), nullable=False)
    theme = db.relationship('Theme', backref=db.backref('tweetimmis', lazy=True))


class Theme(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000), nullable=False, unique=True)

class Tweet_Has_Entity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    entity_text = db.Column(db.String, nullable=False)
    entity_id = db.Column(db.Integer, nullable=False)
    sentiment = db.Column(db.String, nullable=False)

    tweet_id = db.Column(db.Integer, ForeignKey('tweet.id'), nullable=False)
    tweet = db.relationship('Tweet', backref=db.backref('entities', lazy=True))

class Phrase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(1000), nullable=False, unique=True)
    goodness = db.Column(db.String(1000), nullable=False)

    mf = db.Column(db.String(1000), nullable=False)
    stance = db.Column(db.String(1000), nullable=False)

    theme_id = db.Column(db.Integer, ForeignKey('theme.id'), nullable=False)
    theme = db.relationship('Theme', backref=db.backref('phrases', lazy=True))

class PhraseImmi(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(1000), nullable=False, unique=True)
    goodness = db.Column(db.String(1000), nullable=False)

    narrative = db.Column(db.String(1000), nullable=False)
    immi_frame = db.Column(db.String(1000), nullable=False)
    immi_role = db.Column(db.String, nullable=False, default='None')

    # Policy frames
    frame_political = db.Column(db.Boolean, default=False, nullable=False)
    frame_policy = db.Column(db.Boolean, default=False, nullable=False)
    frame_crime = db.Column(db.Boolean, default=False, nullable=False)
    frame_health = db.Column(db.Boolean, default=False, nullable=False)
    frame_security = db.Column(db.Boolean, default=False, nullable=False)
    frame_economic = db.Column(db.Boolean, default=False, nullable=False)

    theme_id = db.Column(db.Integer, ForeignKey('theme.id'), nullable=False)
    theme = db.relationship('Theme', backref=db.backref('phraseimmis', lazy=True))


