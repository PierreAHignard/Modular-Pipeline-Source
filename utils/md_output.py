"""
Module d'export des données d'entraînement sous forme de markdown.
"""

from datetime import datetime
from pathlib import Path

__all__ = [
    "ModelTrainingExporter",
]

class ModelTrainingExporter:
    """Exporte les paramètres et métriques d'entraînement en markdown"""

    def __init__(self, training_config):
        self.config = training_config

        self.test_acc = None
        self.test_loss = None
        self.model_summary = None
        self.model_info = None
        self.report = None

        # Création du répertoire de sortie
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M")
        output_parent = Path(self.config["output_dir_parent"])
        self.output_dir = output_parent / f"{self.config['model_name']}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.assets_dir = self.output_dir / "assets"
        self.assets_dir.mkdir(exist_ok=True)

    def layers_to_markdown_table(self, model):
        """Affiche les couches en table markdown"""

        # S'assurer que le modèle est construit
        if not model.built:
            model.build(input_shape=None)  # ou spécifier une shape d'entrée

        markdown = "| Couche | Type | Output Shape | Params |\n"
        markdown += "|--------|------|--------------|--------|\n"

        for layer in model.layers:
            layer_type = layer.__class__.__name__

            # Vérifier si output_shape existe
            try:
                output_shape = str(layer.output.shape)
            except:
                output_shape = "-"

            params = layer.count_params()
            markdown += f"| {layer.name} | {layer_type} | {output_shape} | {params:,} |\n"

        return markdown

    def add_report(self, report):
        self.report = report

    def set_test_metrics(self, test_loss, test_acc):
        """Définit les métriques de test"""
        self.test_loss = test_loss
        self.test_acc = test_acc
        return self

    def export_to_markdown(self, model):
        """Exporte le rapport complet en markdown"""

        # Pré-formate les métriques de test
        test_acc_str = f"{self.test_acc:.6f}" if self.test_acc is not None else 'N/A'
        test_loss_str = f"{self.test_loss:.6f}" if self.test_loss is not None else 'N/A'

        # Construction du rapport
        md_content = f"""# Rapport d'Entraînement du Modèle

**Modèle**: {self.config['model_name']}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Résumé
{self.config.get('notes', 'N/A')}

---

## Métriques Finales

| Métrique | Valeur |
|----------|--------|
| **Test Accuracy** | **{test_acc_str}** |
| **Test Loss** | **{test_loss_str}** |

---

## Graphiques de Performance

![Métriques d'Entraînement]({"assets/loss_acc.png"}) 

![Matrice de Confusion]({"assets/confusion_matrix.png"})

---

## Paramètres d'Entraînement

| Paramètre | Valeur |
|-----------|--------|
"""

        # Ajoute les paramètres
        for key, value in self.config.items():
            md_content += f"| {key} | {value} |\n"

        # Ajoute les métriques finales
        md_content += f"""

## Model Summary

{self.layers_to_markdown_table(model)}

## Rapport de Classification (test)

{self.report}


## Historique Complet

| Epoch | Loss | Accuracy | Val Loss | Val Accuracy |
|-------|------|----------|----------|--------------|
"""

        for i in range(len(self.metrics_history['epoch'])):
            md_content += f"| {self.metrics_history['epoch'][i]} | "
            md_content += f"{self.metrics_history['loss'][i]:.6f} | "
            md_content += f"{self.metrics_history['accuracy'][i]:.6f} | "
            md_content += f"{self.metrics_history['val_loss'][i]:.6f} | "
            md_content += f"{self.metrics_history['val_accuracy'][i]:.6f} |\n"

        # Sauvegarde le rapport
        md_path = self.output_dir / f"README.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"✅ Rapport exporté: {md_path}")
        return md_path