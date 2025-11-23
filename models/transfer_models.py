import torch
import torch.nn as nn
from torchvision import models


class TransferModel(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=6, pretrained=True):
        super(TransferModel, self).__init__()

        self.model_name = model_name

        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_features, num_classes)

        elif model_name == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

    def freeze_all_except_head(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

        if self.model_name in ['resnet18', 'resnet50']:
            for param in self.base_model.fc.parameters():
                param.requires_grad = True
        elif self.model_name in ['efficientnet_b0', 'mobilenet_v2']:
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True

    def unfreeze_last_layers_resnet(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        for param in self.base_model.fc.parameters():
            param.requires_grad = True

    def get_resnet_param_groups(self, lr_layer4=0.0001, lr_head=0.001):
        return [
            {'params': self.base_model.layer4.parameters(), 'lr': lr_layer4},
            {'params': self.base_model.fc.parameters(), 'lr': lr_head}
        ]