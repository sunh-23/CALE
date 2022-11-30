#coding=utf-8

import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, bottleneck_dim=0, finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.finetune = finetune

        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer

        if bottleneck_dim > 0:
            self.bottleneck = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(backbone.out_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.LeakyReLU(0.2)
            )
            self._features_dim = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features

        self.head = nn.Linear(self._features_dim, num_classes)

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, x):
        f1 = self.pool_layer(self.backbone(x))
        f2 = self.bottleneck(f1)
        predictions = self.head(f2)
        if self.training:
            return predictions, f1, f2
        else:
            return predictions

    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params

