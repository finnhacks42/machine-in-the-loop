from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, PasswordField, IntegerField, TextField,\
    FormField, SelectField, FieldList
from flask_wtf.file import FileField, FileRequired
from wtforms.validators import DataRequired, Length, InputRequired
from wtforms.fields import *


class FileUploadForm(FlaskForm):
    #text = FileField('Text File', validators=[FileRequired()])
    #sbert = FileField('SBERT File', validators=[FileRequired()])
    k = IntegerField('K (# initial clusters, only needed if using K-means)', default=10)
    method = SelectField(choices=[('kmeans', 'K-Means'), ('hdbscan', 'HDBSCAN')])

    submit = SubmitField('Recluster')
    restart = SubmitField('Start from scratch')

class DataForm(FlaskForm):
    theme = SelectField('Theme')
    k = IntegerField('K (# of tweets to show)')
    close = SubmitField('Explore Close Data Points')
    distant = SubmitField('Explore Distant Data Points')

class NewCodeForm(FlaskForm):
    name = StringField('Name', validators=[InputRequired()])
    submit = SubmitField()

class NewPhraseForm(FlaskForm):
    text = StringField('Phrase', validators=[InputRequired()])
    goodness = SelectField(choices=[('good', 'Good'), ('bad', 'Bad')])
    mf = SelectField(choices=[('care/harm', 'Care/Harm'),
                              ('fairness/cheating', 'Fairness/Cheating'),
                              ('loyalty/betrayal', 'Loyalty/Betrayal'),
                              ('purity/degradation', 'Purity/Degradation'),
                              ('authority/subversion', 'Authority/Subversion'),
                              ('liberty/oppression', 'Liberty/Oppression'),
                              ('none', 'None')])
    stance = SelectField(choices=[('pro-vax', 'Pro-Vax'),
                                  ('anti-vax', 'Anti-Vax'),
                                  ('default', 'Neutral')])
    submit = SubmitField()

class EditPhraseForm(FlaskForm):
    goodness = SelectField(choices=[('good', 'Good'), ('bad', 'Bad')])
    mf = SelectField(choices=[('care/harm', 'Care/Harm'),
                              ('fairness/cheating', 'Fairness/Cheating'),
                              ('loyalty/betrayal', 'Loyalty/Betrayal'),
                              ('purity/degradation', 'Purity/Degradation'),
                              ('authority/subversion', 'Authority/Subversion'),
                              ('liberty/oppression', 'Liberty/Oppression'),
                              ('none', 'None')])
    stance = SelectField(choices=[('pro-vax', 'Pro-Vax'),
                                  ('anti-vax', 'Anti-Vax'),
                                  ('default', 'Neutral')])
    submit = SubmitField()

class NewPhraseImmiForm(FlaskForm):
    text = StringField('Phrase', validators=[InputRequired()])
    goodness = SelectField(choices=[('good', 'Good'), ('bad', 'Bad')])
    '''
    policy_frame = SelectMultipleField(choices=[('Capacity_and_Resources', 'Capacity and Resources'),
                                        ('Crime_and_Punishment', 'Crime and Punishment'),
                                        ('Cultural_Identity', 'Cultural Identity'),
                                        ('Economic', 'Economic'),
                                        ('External_Regulation_and_Reputation', 'External Regulation and Reputation'),
                                        ('Fairness_and_Equality', 'Fairness and Equality'),
                                        ('Health_and_Safety', 'Health and Safety'),
                                        ('Legality,_Constitutionality,_Jurisdiction', 'Legality, Constitutionality, Jurisdiction'),
                                        ('Morality_and_Ethics', 'Morality and Ethics'),
                                        ('Policy_Prescription_and_Evaluation', 'Policy Prescription and Evaluation'),
                                        ('Political_Factors_and_Implications', 'Political Factors and Implications'),
                                        ('Public_Sentiment', 'Public Sentiment'),
                                        ('Quality_of_Life', 'Quality of Life'),
                                        ('Security_and_Defense', 'Security and Defense'),
                                        ('None', 'None')
                                       ])
    '''
    immi_frame = SelectField(choices=[('Hero:_Cultural_Diversity', 'Hero: Cultural Diversity'),
                                      ('Hero:_Integration', 'Hero: Integration'),
                                      ('Hero:_Worker', 'Hero: Worker'),
                                      ('Threat:_Fiscal', 'Threat: Fiscal'),
                                      ('Threat:_Jobs', 'Threat: Jobs'),
                                      ('Threat:_National_Cohesion', 'Threat: National Cohesion'),
                                      ('Threat:_Public_Order', 'Threat: Public Order'),
                                      ('Victim:_Discrimination', 'Victim: Discrimination'),
                                      ('Victim:_Global_Economy', 'Victim: Global Economy'),
                                      ('Victim:_Humanitarian', 'Victim: Humanitarian'),
                                      ('Victim:_War', 'Victim: War'),
                                      ('None', 'None')
                                       ])
    immi_role = SelectField(choices=[('Hero', 'Hero'), ('Threat', 'Threat'), ('Victim', 'Victim'), ('None', 'None')])
    narrative = SelectField(choices=[('Episodic', 'Episodic (Concrete)'),
                                  ('Thematic', 'Thematic (Abstract)'),
                                  ('None', 'None')])
    frame_political = BooleanField('frame_political', default=False)
    frame_policy = BooleanField('frame_policy', default=False)
    frame_crime = BooleanField('frame_crime', default=False)
    frame_health = BooleanField('frame_health', default=False)
    frame_security = BooleanField('frame_security', default=False)
    frame_economic = BooleanField('frame_economic', default=False)

    submit = SubmitField()

class EditPhraseImmiForm(FlaskForm):
    goodness = SelectField(choices=[('good', 'Good'), ('bad', 'Bad')])

    immi_frame = SelectField(choices=[('Hero:_Cultural_Diversity', 'Hero: Cultural Diversity'),
                                      ('Hero:_Integration', 'Hero: Integration'),
                                      ('Hero:_Worker', 'Hero: Worker'),
                                      ('Threat:_Fiscal', 'Threat: Fiscal'),
                                      ('Threat:_Jobs', 'Threat: Jobs'),
                                      ('Threat:_National_Cohesion', 'Threat: National Cohesion'),
                                      ('Threat:_Public_Order', 'Threat: Public Order'),
                                      ('Victim:_Discrimination', 'Victim: Discrimination'),
                                      ('Victim:_Global_Economy', 'Victim: Global Economy'),
                                      ('Victim:_Humanitarian', 'Victim: Humanitarian'),
                                      ('Victim:_War', 'Victim: War'),
                                      ('None', 'None')
                                       ])
    immi_role = SelectField(choices=[('Hero', 'Hero'), ('Threat', 'Threat'), ('Victim', 'Victim'), ('None', 'None')])
    narrative = SelectField(choices=[('Episodic', 'Episodic (Concrete)'),
                                  ('Thematic', 'Thematic (Abstract)'),
                                  ('None', 'None')])

    frame_political = BooleanField('frame_political', default=False)
    frame_policy = BooleanField('frame_policy', default=False)
    frame_crime = BooleanField('frame_crime', default=False)
    frame_health = BooleanField('frame_health', default=False)
    frame_security = BooleanField('frame_security', default=False)
    frame_economic = BooleanField('frame_economic', default=False)

    submit = SubmitField()



class ThemeForm(FlaskForm):
    theme = SelectField('Theme')
    visualize = SubmitField('Visualize')
    change_name = SubmitField('Change Name')
    add_phrase = SubmitField('Add Phrase')
    delete = SubmitField('Delete')

class TweetEditForm(FlaskForm):
    theme = SelectField('Theme')
    name = StringField('Name')

class TextQueryForm(FlaskForm):
    query = StringField('Query', validators=[InputRequired()])
    k = IntegerField('K (# of tweets to show)')
    submit = SubmitField('Search')
