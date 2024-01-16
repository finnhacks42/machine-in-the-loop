import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, BertModel
from torch.autograd import Function
import logging
from drail.neuro.nn_model import NeuralNetworks

class FeedForwardClassifier(NeuralNetworks):

    def __init__(self, config, nn_id, use_gpu, output_dim):
        super(FeedForwardClassifier, self).__init__(config, nn_id)
        self.use_gpu = use_gpu
        self.output_dim = output_dim
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("output_dim = {}".format(output_dim))

    def build_architecture(self, rule_template, fe, shared_params={}):
        self.input2hidden = torch.nn.Linear(self.config['n_input'], self.config['n_hidden'])
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(self.config['n_hidden'], self.output_dim)

    def forward(self, x):
        features = self._get_variable(self._get_float_tensor(x['vector']))
        features = self.input2hidden(features)
        features = self.relu(features)
        logits = self.classifier(features)
        probas = F.softmax(logits, dim=1)
        return logits, probas
