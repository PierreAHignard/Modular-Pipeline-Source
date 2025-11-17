"""
Callbacks for model training.
"""

__all__ = [
    "EarlyStopping"
]

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=True):
        """
        Args:
            patience: nombre d'époque sans amélioration avant d'arrêter
            delta: amélioration minimale pour considérer comme une amélioration
            verbose: afficher les messages
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Appeler cette fonction à chaque époque avec la validation loss
        Retourne True si on doit arrêter
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            # Amélioration détectée
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"✓ Amélioration détectée. Best loss: {self.best_loss:.4f}")
        else:
            # Pas d'amélioration
            self.counter += 1
            if self.verbose:
                print(f"⚠ Pas d'amélioration ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping activé après {self.patience} époque(s)")
                return True

        return False
