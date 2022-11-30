#coding=utf-8

import torch
import torch.nn.functional as F

from common.utils.metric import binary_accuracy
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss as base_CDAN

class ConditionalDomainAdversarialLoss(base_CDAN):
    def __init__(self, domain_discriminator, entropy_conditioning=False, randomized=False, num_classes=-1, features_dim=-1, randomized_dim=1024, reduction="mean"):
        super(ConditionalDomainAdversarialLoss, self).__init__(
                domain_discriminator, entropy_conditioning, randomized, num_classes, features_dim, randomized_dim, reduction)
        self.grl.auto_step = False

    def conditional_discriminator(self, logits, features):
        preditions = F.softmax(logits, dim=1).detach()
        h = self.grl(self.map(features, preditions))
        d = self.domain_discriminator(h)
        return d

    def forward(self, logits, features, domain_labels=None):
        d = self.conditional_discriminator(logits, features)

        if self.entropy_conditioning:
            weight = 1. + torch.exp(-entropy(logits))
            weight = weight / torch.sum(weight) * logits.size(0)
        else:
            weight = torch.ones((logits.size(0), 1)).to(logits.device)

        weight = weight.detach()

        batch_size = logits.size(0) // 2
        assert batch_size * 2 == logits.size(0)
        default_domain_labels = torch.cat((
            torch.ones((batch_size, 1)).to(logits.device),
            torch.zeros((batch_size, 1)).to(logits.device)
        ))
        self.domain_discriminator_accuracy = binary_accuracy(d, default_domain_labels)

        if domain_labels is None:
            if self.training:
                return d, self.bce(d, default_domain_labels, weight.view_as(d))
            else:
                return self.bce(d, default_domain_labels, weight.view_as(d))
        else:
            if self.training:
                return d, binary_cross_entropy_with_soft_targets(domain_labels.detach(), d, weight.view_as(d))
            else:
                return binary_cross_entropy_with_soft_targets(domain_labels.detach(), d, weight.view_as(d))


def entropy(logits):
    return -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1, keepdim=True)


def binary_cross_entropy_with_soft_targets(targets, preditions, weight):
    return -torch.mean((
            targets * torch.log(preditions + 1e-6) + \
            (1. - targets) * torch.log(1. - preditions + 1e-6)
        ) * weight)

