"""
Model training script.
"""
import torch
import torch.nn as nn
from tensorflow.core.config.flags import config
from tqdm import tqdm
from model.callbacks import EarlyStopping

__all__ = [
    "Trainer"
]

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.fit.learning_rate,
            weight_decay=config.fit.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.fit.epochs
        )
        self.early_stopping = EarlyStopping(
            patience=config.optimizer.patience,
            delta=config.optimizer.delta,
            verbose=True
        )

        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_epoch(self):
        """Une époque d'entraînement"""
        self.model.train()
        total_loss = 0

        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def validate(self):
        """Validation après chaque époque"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                # Calcul de l'accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        self.history["val_loss"].append(avg_loss)
        self.history["val_acc"].append(accuracy)

        return avg_loss, accuracy

    def fit(self):
        """Boucle d'entraînement complète"""
        for epoch in range(self.config.fit.epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Mise à jour du learning rate
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.config.fit.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Sauvegarde du meilleur modèle
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss
                }
                torch.save(checkpoint, self.config.working_directory / 'checkpoint.tar')
                print("  ✓ Checkpoint sauvegardé")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n✓ Entraînement arrêté à l'époque {epoch + 1}")
                break

        return self.history
