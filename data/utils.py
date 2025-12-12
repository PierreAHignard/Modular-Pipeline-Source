import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

__all__= ['create_balanced_loader']

def create_balanced_loader(dataset, batch_size):
    # Calculer les poids pour chaque échantillon
    class_counts = torch.bincount(torch.tensor([label for _, label in dataset]))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for _, label in dataset]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)