__all__= [
    "unfreeze_last_layers",
    "FocalLoss"
]

import torch
from torch import nn
import torch.nn.functional as F


def unfreeze_last_layers(model, num_layers_to_unfreeze=5):
    """
    Gèle le backbone et dégèle les N dernières couches + classifier.

    Args:
        model: Modèle PyTorch (MobileNetV3, etc.)
        num_layers_to_unfreeze: Nombre de dernières couches à entraîner
    """
    # Geler tout le backbone par défaut
    for param in model.features.parameters():
        param.requires_grad = False

    # Dégeler les N dernières couches
    if num_layers_to_unfreeze > 0:
        for layer in model.features[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Dégeler le classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Afficher les stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
