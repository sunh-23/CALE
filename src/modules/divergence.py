#coding=utf-8

import torch
import torch.nn.functional as F

EPS = 1e-6

class BaseDiv(object):
    def __init__(self, symmetry=False):
        super(BaseDiv, self).__init__()
        self.symmetry = symmetry

    def div_with_logits(self, p_logits, q_logits):
        pass

    def binary_div(self, p_prob, q_prob):
        pass

    def __call__(self, p, q): # q means targets
        assert len(p.shape) == len(q.shape) and len(q.shape) <= 2
        if len(p.shape) == 1:
            p, q = p[:, None], q[:, None]

        if p.shape[-1] == 1:
            if self.symmetry:
                return (self.binary_div(p, q.detach()) + self.binary_div(q.detach(), p)) / 2.
            return self.binary_div(p, q)
        else:
            if self.symmetry:
                return (self.div_with_logits(p, q.detach()) + self.div_with_logits(q.detach(), p)) / 2.
            return self.div_with_logits(p, q)


class KLDiv(BaseDiv):
    def div_with_logits(self, p_logits, q_logits):
        q = F.softmax(q_logits, dim=1)
        qlogq = torch.sum(q * F.log_softmax(q_logits, dim=1), dim=1).mean()
        qlogp = torch.sum(q * F.log_softmax(p_logits, dim=1), dim=1).mean()
        return qlogq - qlogp

    def binary_div(self, p_prob, q_prob):
        qlogq = torch.mean(q_prob * torch.log(q_prob + EPS) + (1. - q_prob) * torch.log(1. - q_prob + EPS))
        qlogp = torch.mean(q_prob * torch.log(p_prob + EPS) + (1. - q_prob) * torch.log(1. - p_prob + EPS))
        return qlogq - qlogp


class CrossEntropy(BaseDiv):
    '''
        soft cross entriopy with weight
    '''
    def __init__(self, symmetry=False, temperature=2.0):
        super(CrossEntropy, self).__init__(symmetry)
        self.temperature = temperature

    def div_with_logits(self, p_logits, q_logits):
        pred = F.softmax(p_logits / self.temperature, dim=1)
        entropy_weight = torch.sum(
                pred * F.log_softmax(p_logits / self.temperature, dim=1), dim=1, keepdim=True)
        entropy_weight = 1 + torch.exp(entropy_weight)
        entropy_weight = (p_logits.size(0) * entropy_weight / entropy_weight.sum()).detach()

        dim_sum = torch.sum(pred * entropy_weight, dim=0, keepdim=True)

        div = -(torch.sum(
            F.softmax(q_logits / self.temperature, dim=1) * \
            F.log_softmax(p_logits / self.temperature, dim=1) / dim_sum,
            dim=1
        ) * entropy_weight).mean()

        return div

    def binary_div(self, p_prob, q_prob):
        div = -torch.mean(q_prob * torch.log(p_prob + EPS) + (1. - q_prob) * torch.log(1. - p_prob + EPS))
        return  div



