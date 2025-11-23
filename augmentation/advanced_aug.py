import torch
from torchvision import transforms
import random
import torch.nn as nn
import torch.nn.functional as F

class BaselineTransform(nn.Module):
    @staticmethod
    def get_transforms(image_size=224):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class BaselineAugmentation:
    @staticmethod
    def get_transforms(image_size=224):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class AdvancedAugmentation:
    @staticmethod
    def get_transforms(image_size=224):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')

        return (1 - self.smoothing) * nll + self.smoothing * (loss / n_classes)


def get_regularized_model(base_model, dropout_rate=0.5):
    class RegularizedWrapper(nn.Module):
        def __init__(self, model, dropout_rate):
            super().__init__()
            self.base_model = model
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            features = self.base_model.base_model.layer4(
                self.base_model.base_model.layer3(
                    self.base_model.base_model.layer2(
                        self.base_model.base_model.layer1(
                            self.base_model.base_model.maxpool(
                                self.base_model.base_model.relu(
                                    self.base_model.base_model.bn1(
                                        self.base_model.base_model.conv1(x)
                                    )
                                )
                            )
                        )
                    )
                )
            )
            features = self.base_model.base_model.avgpool(features)
            features = torch.flatten(features, 1)
            features = self.dropout(features)
            output = self.base_model.base_model.fc(features)
            return output

    return RegularizedWrapper(base_model, dropout_rate)