"""
Model training script with progressive fine-tuning.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from model.callbacks import EarlyStopping

__all__ = [
    "Trainer"
]

class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            config,
            device,
            criterion = nn.CrossEntropyLoss()
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        # Configuration du fine-tuning progressif
        self.progressive_unfreeze = config.fit.progressive_unfreeze
        self.unfreeze_schedule = config.fit.unfreeze_schedule
        self.base_lr = config.fit.learning_rate
        self.classifier_lr_multiplier = config.fit.classifier_lr_multiplier

        # Initialisation : geler le backbone, garder le classificateur entraînable
        if self.progressive_unfreeze:
            self._freeze_backbone()

        # Optimiseur avec learning rates différencié
        self.optimizer = self._create_optimizer()

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
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": []
        }
        self.current_epoch = 0

    def _freeze_backbone(self):
        """Gèle toutes les couches du backbone pré-entraîné"""
        # Pour ResNet/EfficientNet/etc., on gèle tout sauf la dernière couche
        for name, param in self.model.named_parameters():
            if 'fc' not in name and 'classifier' not in name and 'head' not in name:
                param.requires_grad = False
        print("✓ Backbone gelé (seul le classificateur est entraînable)")

    def _get_layer_groups(self):
        """
        Divise le modèle en groupes de couches pour le dégel progressif.
        Adapté pour ResNet, mais peut être modifié pour d'autres architectures.
        """
        # Pour ResNet
        if hasattr(self.model, 'layer4'):
            return [
                self.model.layer4,  # Couches les plus profondes (à dégeler en premier)
                self.model.layer3,
                self.model.layer2,
                self.model.layer1,
                [self.model.conv1, self.model.bn1]  # Couches les plus superficielles
            ]
        # Pour EfficientNet
        elif hasattr(self.model, 'features'):
            features = list(self.model.features.children())
            n = len(features)
            return [
                features[int(n*0.75):],  # 25% supérieur
                features[int(n*0.5):int(n*0.75)],  # 25% suivant
                features[int(n*0.25):int(n*0.5)],  # 25% suivant
                features[:int(n*0.25)]  # 25% inférieur
            ]
        else:
            # Fallback générique
            return [list(self.model.children())]

    def _unfreeze_layer_group(self, group_idx):
        """Dégèle un groupe de couches spécifique"""
        layer_groups = self._get_layer_groups()

        if group_idx < len(layer_groups):
            group = layer_groups[group_idx]
            if isinstance(group, list):
                for layer in group:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                for param in group.parameters():
                    param.requires_grad = True

            print(f"✓ Groupe de couches {group_idx + 1} dégelé")

            # Recréer l'optimiseur avec les nouveaux paramètres
            self.optimizer = self._create_optimizer()

            # Recréer le scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.fit.epochs - self.current_epoch
            )

    def _create_optimizer(self):
        """
        Crée l'optimiseur avec des learning rates différenciés.
        Le classificateur a un LR plus élevé, les couches pré-entraînées un LR plus faible.
        """
        # Séparer les paramètres du classificateur et du backbone
        classifier_params = []
        backbone_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'fc' in name or 'classifier' in name or 'head' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)

        # Créer des groupes de paramètres avec des LR différents
        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.base_lr
            })

        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': self.base_lr * self.classifier_lr_multiplier
            })

        return torch.optim.Adam(
            param_groups,
            weight_decay=self.config.fit.weight_decay
        )

    def _check_unfreeze_schedule(self):
        """Vérifie si des couches doivent être dégelées à cette époque"""
        if not self.progressive_unfreeze:
            return

        for idx, epoch_threshold in enumerate(self.unfreeze_schedule):
            if self.current_epoch == epoch_threshold:
                self._unfreeze_layer_group(idx)
                print(f"  → Learning rate actuel du backbone: {self.optimizer.param_groups[0]['lr']:.2e}")
                if len(self.optimizer.param_groups) > 1:
                    print(f"  → Learning rate actuel du classificateur: {self.optimizer.param_groups[1]['lr']:.2e}")

    def train_epoch(self):
        """Une époque d'entraînement"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

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

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100 * correct / total

        self.history["train_loss"].append(avg_loss)
        self.history["train_acc"].append(train_acc)

        return avg_loss, train_acc

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

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        self.history["val_loss"].append(avg_loss)
        self.history["val_acc"].append(accuracy)

        return avg_loss, accuracy

    def fit(self):
        """Boucle d'entraînement complète avec fine-tuning progressif"""
        for epoch in range(self.config.fit.epochs):
            self.current_epoch = epoch

            # Vérifier si des couches doivent être dégelées
            self._check_unfreeze_schedule()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Mise à jour du learning rate
            self.scheduler.step()

            # Enregistrer le learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["lr"].append(current_lr)

            print(f"Epoch {epoch+1}/{self.config.fit.epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")

            # Sauvegarde du meilleur modèle
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss,
                    'accuracy': val_acc
                }
                torch.save(checkpoint, self.config.working_directory / 'checkpoint.tar')
                print("  ✓ Checkpoint sauvegardé")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n✓ Entraînement arrêté à l'époque {epoch + 1}")
                break

        return self.history
