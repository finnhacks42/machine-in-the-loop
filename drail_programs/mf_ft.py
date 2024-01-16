import json
from drail.features.feature_extractor import FeatureExtractor

class MF_ft(FeatureExtractor):

    def __init__(self, tweets_f, entities_f, themes_f, debug=False):
        self.tweets_f = tweets_f
        self.entities_f = entities_f
        self.themes_f = themes_f

    def build(self):
        self.tweets = json.load(open(self.tweets_f))
        if self.entities_f is not None:
            self.entities = json.load(open(self.entities_f))

        self.morality2idx = {
                'moral': 0,
                'non-moral': 1
        }

        self.mf2idx = {
            'authority/subversion': 0,
            'care/harm': 1,
            'fairness/cheating': 2,
            'liberty/oppression': 3,
            'loyalty/betrayal': 4,
            'none': 5,
            'purity/degradation': 6
        }

        self.sent2idx = {
            'neg': 0,
            'pos': 1
        }

        self.role2idx = {
            'actor': 0,
            'target': 1
        }

        self.stance2idx = {
            'pro-vax': 0,
            'anti-vax': 1,
            'default': 2
        }

        self.immirole2idx = {
            'Hero': 0,
            'Threat': 1,
            'Victim': 2,
            'None': 3
        }

        self.immiframe2idx = {
            'Victim_Global_Economy': 0,
            'Victim_Humanitarian': 1,
            'Victim_War': 2,
            'Victim_Discrimination': 3,
            'Hero_Cultural_Diversity': 4,
            'Hero_Integration': 5,
            'Hero_Worker': 6,
            'Threat_Jobs': 7,
            'Threat_Public_Order': 8,
            'Threat_Fiscal': 9,
            'Threat_National_Cohesion': 10,
            'None': 11
        }

        self.narrative2idx = {
            'Episodic': 0,
            'Thematic': 1,
            'None': 2
        }

        theme_idx = 0; self.theme2idx = {}
        with open(self.themes_f) as fp:
            for line in fp:
                self.theme2idx[line.strip()] = theme_idx
                theme_idx += 1


    def tweet_bert(self, rule_grd):
        tweet_id = rule_grd.get_body_predicates('IsTweet')[0]['arguments'][0]
        encoded_inputs = self.tweets[tweet_id]
        if (not isinstance(encoded_inputs['input_ids'], list) or
            not isinstance(encoded_inputs['token_type_ids'], list) or
            not isinstance(encoded_inputs['attention_mask'], list)):

            print(type(encoded_inputs['input_ids']), encoded_inputs['input_ids'])
            print(type(encoded_inputs['token_type_ids']), encoded_inputs['token_type_ids'])
            print(type(encoded_inputs['attention_mask']), encoded_inputs['attention_mask'])
            exit()
        return (encoded_inputs['input_ids'], encoded_inputs['token_type_ids'], encoded_inputs['attention_mask'])

    def tweet_entity_bert(self, rule_grd):
        entity_id = rule_grd.get_body_predicates('HasEntity')[0]['arguments'][1]
        encoded_inputs = self.entities[entity_id]
        return (encoded_inputs['input_ids'], encoded_inputs['token_type_ids'], encoded_inputs['attention_mask'])

    def role_1hot(self, rule_grd):
        role_id = rule_grd.get_body_predicates('HasRole')[-1]['arguments'][-1]
        vect = [0.0] * len(self.role2idx)
        vect[self.role2idx[role_id]] = 1.0
        return vect

    def sentiment_1hot(self, rule_grd):
        sent_id = rule_grd.get_body_predicates('HasSentiment')[-1]['arguments'][-1]
        vect = [0.0] * len(self.sent2idx)
        vect[self.sent2idx[sent_id]] = 1.0
        return vect

    def stance_1hot(self, rule_grd):
        stance_id = rule_grd.get_body_predicates('HasStance')[-1]['arguments'][-1]
        vect = [0.0, 0.0, 0.0]
        if stance_id == "neutral":
            stance_id = "default"
        vect[self.stance2idx[stance_id]] = 1.0
        return vect

    def morality_1hot(self, rule_grd):
        moral_id = rule_grd.get_body_predicates('HasMorality')[-1]['arguments'][-1]
        if moral_id == 'None':
            moral_id = 'non-moral'
        vect = [0.0] * len(self.morality2idx)
        vect[self.morality2idx[moral_id]] = 1.0
        return vect

    def mf_1hot(self, rule_grd):
        moral_id = rule_grd.get_body_predicates('HasMoralFoundation')[-1]['arguments'][-1]
        if moral_id == 'None':
            moral_id = 'none'
        vect = [0.0] * len(self.mf2idx)
        vect[self.mf2idx[moral_id]] = 1.0
        return vect

    def narrative_1hot(self, rule_grd):
        #print(rule_grd)
        narrative_id = rule_grd.get_body_predicates('HasNarrative')[-1]['arguments'][-1]
        vect = [0.0] * len(self.narrative2idx)
        vect[self.narrative2idx[narrative_id]] = 1.0
        return vect

    def immi_role_1hot(self, rule_grd):
        _id = rule_grd.get_body_predicates('HasImmiRole')[-1]['arguments'][-1]
        vect = [0.0] * len(self.immirole2idx)
        vect[self.immirole2idx[_id]] = 1.0
        return vect

    def immi_frame_1hot(self, rule_grd):
        _id = rule_grd.get_body_predicates('HasImmiFrame')[-1]['arguments'][-1]
        vect = [0.0] * len(self.immiframe2idx)
        vect[self.immiframe2idx[_id]] = 1.0
        return vect

    def frame_vect(self, rule_grd):
        vect = [0.0] * 6
        for i, pred in enumerate(['Political', 'Policy', 'Crime', 'Health', 'Security', 'Economic']):
            pred_name = "HasFrame{}".format(pred)
            _id = rule_grd.get_body_predicates(pred_name)[-1]['arguments'][-1]
            if int(_id) == 1:
                vect[i] = 1.0
        return vect

    def arg_1hot(self, rule_grd):
        arg_id = rule_grd.get_body_predicates("IsArgument")[-1]['arguments'][-1]
        vect = [0.0] * len(self.theme2idx)
        vect[self.theme2idx[arg_id]] = 1.0
        return vect

    def extract_multiclass_head(self, rule_grd):
        head = rule_grd.get_head_predicate()
        label = head['arguments'][-1]
        if head['name'] == 'HasRole':
            return self.role2idx[label]
        elif head['name'] == 'HasSentiment':
            return self.sent2idx[label]
        elif head['name'] == 'HasMorality':
            return self.morality2idx[label]
        elif head['name'] == 'HasMoralFoundation':
            return self.mf2idx[label]
        elif head['name'] == 'MentionsArgument':
            return self.theme2idx[label]
        elif head['name'] == 'HasStance':
            return self.stance2idx[label]
        elif head['name'] == 'HasImmiRole':
            return self.immirole2idx[label]
        elif head['name'] == 'HasImmiFrame':
            return self.immiframe2idx[label]
        elif head['name'] == 'HasNarrative':
            return self.narrative2idx[label]
        elif head['name'].startswith('HasFrame'):
            return int(label)
        else:
            return int(label)
