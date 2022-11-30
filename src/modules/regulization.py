#coding=utf-8

import contextlib
import torch
from torch import nn
import torch.nn.functional as F
import modules.divergence as D


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


class AT(object): # Adversarial Training
    def __init__(self, classifier, divergence, eps=1.0):
        super(AT, self).__init__()
        self.classifier = classifier
        self.divergence = divergence
        self.eps = eps

    def l2_normalize(self, v):
        v /= (torch.max(torch.abs(v), dim=1, keepdim=True)[0] + 1e-6)
        v /= (torch.norm(v, dim=1, keepdim=True) + 1e-8)
        return v

    def __call__(self, inputs, q_logits):
        d = torch.randn(inputs.shape).to(inputs.device)
        d = self.l2_normalize(d)
        inputs = inputs.clone().detach()

        with _disable_tracking_bn_stats(self.classifier):
            inputs.requires_grad_()

            p_logits = self.classifier(inputs)
            div = self.divergence(p_logits, q_logits.detach())

            self.classifier.zero_grad()
            div.backward()
            d = self.l2_normalize(inputs.grad.detach())

        r = self.eps * d

        return r.detach()


class VAT(object): # Virtual Adversarial Training
    def __init__(self, classifier, divergence, xi=10., eps=1.0, K=3):
        super(VAT, self).__init__()
        self.classifier = classifier
        self.divergence = divergence
        self.xi = xi
        self.eps = eps
        self.K = K

    def l2_normalize(self, v):
        v /= (torch.max(torch.abs(v), dim=1, keepdim=True)[0] + 1e-6)
        v /= (torch.norm(v, dim=1, keepdim=True) + 1e-8)
        return v

    def __call__(self, inputs, q_logits):
        d = torch.randn(inputs.shape).to(inputs.device)
        d = self.l2_normalize(d)
        inputs = inputs.detach()

        with _disable_tracking_bn_stats(self.classifier):
            for _ in range(self.K):
                d.requires_grad_()

                p_logits = self.classifier(inputs + self.xi * d)
                div = self.divergence(p_logits, q_logits.detach())

                self.classifier.zero_grad()
                div.backward()
                d = self.l2_normalize(d.grad.detach())

        r = self.eps * d

        return r.detach()

class ConditionalDiscriminator(nn.Module):
    def __init__(self, bottleneck, classifier, conditional_discriminator):
        super(ConditionalDiscriminator, self).__init__()
        self.bottleneck = bottleneck
        self.classifier = classifier
        self.conditional_discriminator = conditional_discriminator

    def forward(self, features):
        features = self.bottleneck(features)
        cls_logits = self.classifier(features)
        d = self.conditional_discriminator(cls_logits, features)
        return d

# Regulization: Cooperative and Adversarial Learning
class CooperativeAdversarialLearning(object):
    def __init__(self, bottleneck, classifier, domain_adv, divergence, args):
        self.bottleneck = bottleneck
        self.classifier = classifier
        self.domain_adv = domain_adv
        self.conditional_discriminator = ConditionalDiscriminator(
            self.bottleneck, self.classifier, self.domain_adv.conditional_discriminator)

        self.divergence = divergence
        if args.data == "VisDA2017":
            self.loss_f = D.KLDiv()
        else:
            self.loss_f = divergence

        self.adversarial_cls_generator = AT(
            nn.Sequential(
                self.bottleneck,
                self.classifier
            ), self.divergence, eps=args.eps
        )

        self.cooperative_cls_generator = AT(
            nn.Sequential(
                self.bottleneck,
                self.classifier
            ), self.divergence, eps=-1. * args.eps
        )

        self.adversarial_dis_generator = AT(
            self.conditional_discriminator, self.divergence, args.eps)

        self.cooperative_dis_generator = AT(
            self.conditional_discriminator, self.divergence, -1. * args.eps)

    def generate_discrminable_samples(self, features, q_logits):
        r_coo = self.cooperative_cls_generator(features, q_logits)
        return features + r_coo.detach()

    def generate_non_discrminable_samples(self, features, q_logits):
        r_adv = self.adversarial_cls_generator(features, q_logits)
        return features + r_adv.detach()

    def generate_transferable_samples(self, features, q_logits):
        r_coo = self.cooperative_dis_generator(features, q_logits)
        return features + r_coo.detach()

    def generate_non_transferable_samples(self, features, q_logits):
        r_adv = self.adversarial_dis_generator(features, q_logits)
        return features + r_adv.detach()

    def regulization_loss(self, features, \
            cls_logits, dis_logits, at_cls_logits, at_dis_logits):
        # dis_logits are not real logits, they are applied sigmoid
        disc_features     = self.generate_discrminable_samples(features, at_cls_logits)
        non_disc_features = self.generate_non_discrminable_samples(features, at_cls_logits)
        tran_features     = self.generate_transferable_samples(features, at_dis_logits)
        non_tran_features = self.generate_non_transferable_samples(features, at_dis_logits)

        # cls_regulization: non_disc, tran
        cls_reg_features = torch.cat((non_disc_features, tran_features), dim=0)
        cls_output = self.classifier(self.bottleneck(cls_reg_features))
        cls_reg_loss = self.loss_f(cls_output, torch.cat((cls_logits, cls_logits), dim=0))

        # dis_regulization: non_tran, disc
        dis_reg_features = torch.cat((non_tran_features, disc_features), dim=0)
        dis_output = self.domain_adv.conditional_discriminator(
            cls_output, self.bottleneck(dis_reg_features))
        dis_reg_loss = self.loss_f(dis_output, torch.cat((dis_logits, dis_logits), dim=0))

        regulization_loss = cls_reg_loss + dis_reg_loss

        return regulization_loss

