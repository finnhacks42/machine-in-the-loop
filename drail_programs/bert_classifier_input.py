import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, BertModel
from torch.autograd import Function
import logging
from drail.neuro.nn_model import NeuralNetworks

class RevGradFun(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None

class RevGrad(torch.nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False).cuda()

    def forward(self, input_):
        return RevGradFun.apply(input_, self._alpha)

class BertClassifier(NeuralNetworks):

    def __init__(self, config, nn_id, use_gpu, output_dim):
        super(BertClassifier, self).__init__(config, nn_id)
        self.use_gpu = use_gpu
        self.output_dim = output_dim
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("output_dim = {}".format(output_dim))

    def build_architecture(self, rule_template, fe, shared_params={}):
        bert_config = AutoConfig.from_pretrained(self.config['bert_model'])
        self.bert_model = AutoModel.from_pretrained(self.config['bert_model'], add_pooling_layer=True)
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(bert_config.hidden_size + self.config['n_input'], self.output_dim)
        self.rev_grad = RevGrad()
        self.adversarial = False

    def forward(self, x):
        input_ids = []; input_mask = []; segment_ids = []
        for elem in x['input']:
            input_ids.append(elem[0][0])
            segment_ids.append(elem[0][1])
            input_mask.append(elem[0][2])

        input_ids = self._get_variable(self._get_long_tensor(input_ids))
        input_mask = self._get_variable(self._get_long_tensor(input_mask))
        segment_ids = self._get_variable(self._get_long_tensor(segment_ids))

        outputs = self.bert_model(input_ids, attention_mask=input_mask,
                                 token_type_ids=segment_ids,
                                 position_ids=None, head_mask=None)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        features = self._get_variable(self._get_float_tensor(x['vector']))
        pooled_output = torch.cat([pooled_output, features], dim=1)


        if self.adversarial:
            pooled_output = self.rev_grad(pooled_output)

        logits = self.classifier(pooled_output)
        probas = F.softmax(logits, dim=1)
        return logits, probas
