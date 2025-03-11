import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from PIL import Image
import cv2
import random
import json
from datetime import datetime

# Importer les classes du modèle
from card_classifier import CardClassifier, ColorEmphasisTransform, EnhancedColorAttention, GradCAM


class CardDetectionTester:
    """
    Classe pour évaluer les performances du détecteur de cartons sur un ensemble de test indépendant.
    """

    def __init__(self, model_path, test_data_dir=None, validation_data_dir=None,
                 split_validation=False, test_ratio=0.5, batch_size=16, save_dir='test_results'):
        """
        Initialise le testeur avec le modèle entraîné et configure les données de test.

        Args:
            model_path (str): Chemin vers le modèle entraîné
            test_data_dir (str, optional): Répertoire des données de test. Si None et split_validation=True,
                                          utilisera validation_data_dir pour créer un ensemble de test.
            validation_data_dir (str, optional): Répertoire des données de validation, utilisé si split_validation=True.
            split_validation (bool): Si True, divisera l'ensemble de validation pour créer un ensemble de test.
            test_ratio (float): Proportion de l'ensemble de validation à utiliser comme test si split_validation=True.
            batch_size (int): Taille des lots pour l'évaluation.
            save_dir (str): Répertoire où sauvegarder les résultats des tests.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.batch_size = batch_size
        self.save_dir = save_dir

        # Créer le répertoire de sauvegarde s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)

        # Charger le modèle
        self.model = self._load_model()

        # Configurer le prétraitement d'images
        self.transform = transforms.Compose([
            ColorEmphasisTransform(saturation_factor=1.8, red_boost=1.3, yellow_boost=1.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Préparer les données de test
        if split_validation and validation_data_dir is not None:
            self.test_loader, self.class_names = self._split_validation_set(validation_data_dir, test_ratio)
        elif test_data_dir is not None:
            self.test_loader, self.class_names = self._load_test_data(test_data_dir)
        else:
            raise ValueError(
                "Vous devez fournir soit un répertoire de test, soit un répertoire de validation à diviser.")

        # Initialiser GradCAM pour les visualisations
        self.grad_cam = GradCAM(self.model)

    def _load_model(self):
        """Charge le modèle entraîné et le place en mode évaluation."""
        print(f"Chargement du modèle depuis {self.model_path}")
        model = CardClassifier().to(self.device)

        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Modèle chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            raise

        model.eval()
        return model

    def _load_test_data(self, test_data_dir):
        """Charge les données de test depuis un répertoire dédié."""
        print(f"Chargement des données de test depuis {test_data_dir}")

        test_dataset = datasets.ImageFolder(
            test_data_dir,
            transform=self.transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

        return test_loader, test_dataset.classes

    def _split_validation_set(self, validation_data_dir, test_ratio=0.5):
        """Divise l'ensemble de validation pour créer un ensemble de test."""
        print(f"Division de l'ensemble de validation {validation_data_dir} pour créer un ensemble de test")

        # Charger l'ensemble de validation complet
        validation_dataset = datasets.ImageFolder(
            validation_data_dir,
            transform=self.transform
        )

        # Obtenir le nombre total d'échantillons et calculer la taille de l'ensemble de test
        total_samples = len(validation_dataset)
        test_size = int(total_samples * test_ratio)
        val_size = total_samples - test_size

        print(f"Ensemble de validation original: {total_samples} échantillons")
        print(f"Nouvel ensemble de validation: {val_size} échantillons")
        print(f"Nouvel ensemble de test: {test_size} échantillons")

        # Diviser le dataset en gardant une distribution équilibrée des classes
        # Pour cela, nous divisons chaque classe séparément

        class_indices = {}
        for idx, (_, label) in enumerate(validation_dataset.samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        test_indices = []
        for label, indices in class_indices.items():
            # Mélanger les indices pour cette classe
            random.shuffle(indices)
            # Prendre test_ratio % des indices pour l'ensemble de test
            test_indices_for_class = indices[:int(len(indices) * test_ratio)]
            test_indices.extend(test_indices_for_class)

        # Créer un sous-ensemble pour l'ensemble de test
        test_subset = Subset(validation_dataset, test_indices)

        # Créer le DataLoader pour l'ensemble de test
        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

        return test_loader, validation_dataset.classes

    def run_test(self):
        """Exécute l'évaluation complète sur l'ensemble de test."""
        print("Démarrage de l'évaluation sur l'ensemble de test...")

        # Collecter les prédictions et les étiquettes
        all_preds = []
        all_probs = []
        all_labels = []
        test_images = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Si c'est le premier lot, sauvegarder quelques images pour visualisation
                if batch_idx == 0:
                    for i in range(min(5, len(inputs))):
                        test_images.append((inputs[i], labels[i]))

                # Forward pass
                outputs, _, _, _, _ = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)

                # Obtenir les prédictions
                _, predicted = torch.max(outputs, 1)

                # Collecter les résultats
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculer les métriques
        self.compute_metrics(all_preds, all_labels, all_probs)

        # Générer des visualisations
        self.visualize_results(test_images, all_preds, all_labels, all_probs)

        print("Évaluation terminée. Résultats sauvegardés dans", self.save_dir)

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'report': classification_report(all_labels, all_preds, target_names=self.class_names)
        }

    def compute_metrics(self, predictions, labels, probabilities):
        """Calcule et sauvegarde les métriques de performance."""
        # Précision globale
        accuracy = accuracy_score(labels, predictions)

        # Matrice de confusion
        cm = confusion_matrix(labels, predictions)

        # Rapport de classification détaillé
        report = classification_report(labels, predictions, target_names=self.class_names, output_dict=True)

        # Précision, rappel et F1-score par classe
        precision, recall, f1, support = precision_recall_fscore_support(labels, predictions)

        # Sauvegarder les métriques dans un fichier
        metrics = {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist(),
            'class_names': self.class_names
        }

        # Sauvegarder au format JSON
        with open(os.path.join(self.save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Afficher les résultats
        print(f"Précision globale: {accuracy:.4f}")
        print("\nMatrice de confusion:")
        print(cm)
        print("\nRapport de classification:")
        print(classification_report(labels, predictions, target_names=self.class_names))

        # Visualiser la matrice de confusion
        self.plot_confusion_matrix(cm)

    def plot_confusion_matrix(self, cm):
        """Génère et sauvegarde une visualisation de la matrice de confusion."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité terrain')
        plt.title('Matrice de confusion sur l\'ensemble de test')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()

    def visualize_results(self, test_images, predictions, labels, probabilities):
        """Génère des visualisations des résultats du test."""
        self.visualize_predictions(test_images)
        self.visualize_error_cases(test_images, predictions, labels)

    def visualize_predictions(self, test_images):
        """Visualise les prédictions du modèle sur quelques images de test avec Grad-CAM."""
        plt.figure(figsize=(20, 4 * len(test_images)))

        for i, (image, label) in enumerate(test_images):
            # Prédire la classe
            with torch.set_grad_enabled(True):
                outputs, _, _, _, _ = self.model(image.unsqueeze(0))
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                # Générer la carte de chaleur Grad-CAM
                cam = self.grad_cam.generate_cam(image.unsqueeze(0), predicted.item())

                # Convertir l'image en numpy array
                img = image.cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                # Visualisation Grad-CAM
                cam_img = self.grad_cam.overlay_heatmap(cam, (img * 255).astype(np.uint8))

                # Afficher l'image originale
                plt.subplot(len(test_images), 2, 2 * i + 1)
                plt.imshow(img)
                true_class = self.class_names[label.item()]
                pred_class = self.class_names[predicted.item()]
                plt.title(f"Vraie: {true_class}\nPrédiction: {pred_class} ({confidence.item():.2f})")
                plt.axis('off')

                # Afficher la visualisation Grad-CAM
                plt.subplot(len(test_images), 2, 2 * i + 2)
                plt.imshow(cam_img)
                plt.title("Zones d'attention (Grad-CAM)")
                plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'predictions_visualization.png'))
        plt.close()

    def visualize_error_cases(self, all_images, predictions, labels):
        """Visualise les cas d'erreur du modèle."""
        # Cette fonction sera appelée si nécessaire pour analyser des cas spécifiques
        # Pour l'instant, nous utilisons les visualisations standard
        pass

    def save_model_summary(self):
        """Sauvegarde un résumé de l'architecture du modèle et des paramètres."""
        # Compter le nombre de paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Créer un résumé
        summary = {
            'model_path': self.model_path,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': str(self.device)
        }

        # Sauvegarder au format JSON
        with open(os.path.join(self.save_dir, 'model_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)


# Exemple d'utilisation
if __name__ == "__main__":
    # Pour diviser l'ensemble de validation existant
    tester = CardDetectionTester(
        model_path="card_model_color_enhanced.pth",
        validation_data_dir="dataset/validation",
        split_validation=True,
        test_ratio=0.5,
        save_dir="test_results"
    )

    # Exécuter les tests
    results = tester.run_test()

    # Sauvegarder un résumé du modèle
    tester.save_model_summary()