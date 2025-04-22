import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from card_classifier import CardDetector, CardClassifier
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model_detailed(model_path, val_dir, batch_size=16):
    """
    Évalue en détail un modèle de classification de cartons sur l'ensemble de validation.

    Args:
        model_path (str): Chemin vers le modèle sauvegardé
        val_dir (str): Chemin vers le répertoire de validation
        batch_size (int): Taille du batch pour l'évaluation

    Returns:
        dict: Dictionnaire contenant toutes les métriques d'évaluation
    """
    print(f"Évaluation détaillée du modèle: {model_path}")

    # Créer le détecteur avec le modèle chargé
    detector = CardDetector(model_path)

    # Préparer les transformations pour la validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Charger le dataset de validation
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Récupérer les noms des classes
    class_names = val_dataset.classes
    print(f"Classes: {class_names}")

    # Passer en mode évaluation
    detector.model.eval()

    # Initialiser les variables pour collecter les prédictions et les vraies étiquettes
    all_preds = []
    all_labels = []
    all_probs = []  # Pour les scores de confiance

    # Évaluer le modèle
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(detector.device), labels.to(detector.device)
            outputs, _, _, _, _ = detector.model(inputs)

            # Calculer les probabilités avec softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Obtenir les prédictions
            _, predicted = torch.max(outputs, 1)

            # Stocker les résultats
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convertir en arrays numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculer les métriques
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Calculer l'accuracy par classe
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        # Sélectionner seulement les instances de cette classe
        mask = (all_labels == i)
        if np.sum(mask) > 0:  # Éviter la division par zéro
            class_accuracy[class_name] = np.mean(all_preds[mask] == all_labels[mask])
        else:
            class_accuracy[class_name] = 0.0

    # Obtenir un rapport de classification détaillé
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Créer un DataFrame pour une meilleure visualisation
    report_df = pd.DataFrame(report).transpose()

    # Calculer l'intervalle de confiance à 95% pour l'accuracy
    n = len(all_labels)
    confidence_interval = 1.96 * np.sqrt((accuracy * (1 - accuracy)) / n)

    # Créer un dictionnaire avec tous les résultats
    results = {
        'accuracy': accuracy,
        'accuracy_percent': accuracy * 100,
        'confidence_interval': confidence_interval,
        'class_accuracy': class_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'classification_report_df': report_df,
        'n_samples': n,
        'class_names': class_names
    }

    return results


def print_evaluation_results(results, paper_accuracy=79.90):
    """
    Affiche les résultats d'évaluation de manière claire et informative.

    Args:
        results (dict): Dictionnaire contenant les résultats d'évaluation
        paper_accuracy (float): Précision rapportée dans l'article pour comparaison
    """
    print("\n" + "=" * 80)
    print(f"RÉSULTATS D'ÉVALUATION DU MODÈLE")
    print("=" * 80)

    accuracy = results['accuracy_percent']
    ci = results['confidence_interval'] * 100

    print(f"\nPrécision globale: {accuracy:.2f}% ± {ci:.2f}%")
    print(f"Nombre d'échantillons évalués: {results['n_samples']}")

    # Comparer avec la référence du papier
    diff = accuracy - paper_accuracy
    print(f"\nComparaison avec le papier de référence:")
    print(f"- Précision rapportée dans le papier: {paper_accuracy:.2f}%")
    print(f"- Différence: {diff:+.2f}% ({diff / paper_accuracy * 100:+.2f}% de la valeur de référence)")

    # Afficher la précision par classe
    print("\nPrécision par classe:")
    for class_name, acc in results['class_accuracy'].items():
        print(f"- {class_name}: {acc * 100:.2f}%")

    # Afficher un tableau détaillé des métriques par classe
    print("\nMétriques détaillées par classe:")
    report_df = results['classification_report_df']

    # Formatter le DataFrame pour un affichage plus clair
    formatted_df = report_df.copy()
    for col in ['precision', 'recall', 'f1-score']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x * 100:.2f}%" if isinstance(x, (int, float)) else x)

    print(formatted_df)

    print("\n" + "=" * 80)


def visualize_evaluation_results(results, paper_accuracy=79.90, save_path="model_evaluation.png"):
    """
    Visualise les résultats d'évaluation du modèle.

    Args:
        results (dict): Dictionnaire contenant les résultats d'évaluation
        paper_accuracy (float): Précision rapportée dans l'article pour comparaison
        save_path (str): Chemin pour sauvegarder la visualisation
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Graphique de la précision globale et par classe
    ax = axes[0]

    # Préparer les données
    class_names = list(results['class_accuracy'].keys())
    class_accuracies = [results['class_accuracy'][cls] * 100 for cls in class_names]

    # Ajouter la précision globale
    all_names = ['Global'] + class_names
    all_accuracies = [results['accuracy_percent']] + class_accuracies

    # Ajouter la référence du papier
    all_names.append('Papier (réf.)')
    all_accuracies.append(paper_accuracy)

    # Créer les barres
    bars = ax.bar(all_names, all_accuracies, color=['blue', 'green', 'red', 'purple'])

    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom')

    # Configurer le graphique
    ax.set_ylim([0, 105])
    ax.set_ylabel('Précision (%)')
    ax.set_title('Comparaison des précisions')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Matrice de confusion heatmap
    ax = axes[1]
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=results['class_names'],
                yticklabels=results['class_names'],
                ax=ax)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')
    ax.set_title('Matrice de confusion')

    # Finalisation et sauvegarde
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Visualisation sauvegardée dans '{save_path}'")
    plt.show()



def evaluate_card_model(model_path, val_dir, batch_size=16, paper_accuracy=79.90):
    """
    Fonction principale pour évaluer complètement le modèle de classification de cartons.

    Args:
        model_path (str): Chemin vers le modèle sauvegardé
        val_dir (str): Chemin vers le répertoire de validation
        batch_size (int): Taille du batch pour l'évaluation
        paper_accuracy (float): Précision rapportée dans l'article pour comparaison
    """
    # Évaluer le modèle
    results = evaluate_model_detailed(model_path, val_dir, batch_size)

    # Afficher les résultats
    print_evaluation_results(results, paper_accuracy)

    # Visualiser les résultats
    visualize_evaluation_results(results, paper_accuracy)

    return results


def evaluate_on_test_set(model_path, test_dir, batch_size=16, paper_accuracy=79.90):
    """
    Évalue le modèle sur un ensemble de test avec visualisations complètes.

    Args:
        model_path (str): Chemin vers le modèle sauvegardé
        test_dir (str): Chemin vers le répertoire de test
        batch_size (int): Taille du batch pour l'évaluation
        paper_accuracy (float): Précision rapportée dans l'article pour comparaison
    """

    print(f"Évaluation sur l'ensemble de test: {test_dir}")

    # Charger le détecteur avec le modèle entraîné
    detector = CardDetector(model_path)

    # Préparation des transformations (identiques à celles de validation, SANS augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Vérifier si le répertoire existe
    if not os.path.exists(test_dir):
        print(f"ERREUR: Le répertoire de test '{test_dir}' n'existe pas!")
        return None

    # Charger le dataset de test
    try:
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f"Ensemble de test chargé: {len(test_dataset)} images")
        print(f"Classes: {test_dataset.classes}")

        # Passer en mode évaluation
        detector.model.eval()

        # Variables pour collecter les prédictions
        all_preds = []
        all_labels = []
        all_probs = []

        # Évaluer le modèle
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(detector.device), labels.to(detector.device)
                outputs, _, _, _, _ = detector.model(inputs)

                # Calculer les probabilités avec softmax
                probs = torch.nn.functional.softmax(outputs, dim=1)

                # Obtenir les prédictions
                _, predicted = torch.max(outputs, 1)

                # Stocker les résultats
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Convertir en arrays numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculer les métriques
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        # Calculer l'accuracy par classe
        class_accuracy = {}
        for i, class_name in enumerate(test_dataset.classes):
            # Sélectionner seulement les instances de cette classe
            mask = (all_labels == i)
            if np.sum(mask) > 0:  # Éviter la division par zéro
                class_accuracy[class_name] = np.mean(all_preds[mask] == all_labels[mask])
            else:
                class_accuracy[class_name] = 0.0

        # Calculer l'intervalle de confiance à 95% pour l'accuracy
        n = len(all_labels)
        confidence_interval = 1.96 * np.sqrt((accuracy * (1 - accuracy)) / n)

        # Obtenir un rapport de classification détaillé
        report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Afficher les résultats
        print("\n" + "=" * 80)
        print(f"RÉSULTATS SUR L'ENSEMBLE DE TEST")
        print("=" * 80)
        print(f"\nPrécision globale: {accuracy * 100:.2f}% ± {confidence_interval * 100:.2f}%")
        print(f"Nombre d'échantillons évalués: {n}")

        # Comparer avec la référence du papier
        diff = accuracy * 100 - paper_accuracy
        print(f"\nComparaison avec le papier de référence:")
        print(f"- Précision rapportée dans le papier: {paper_accuracy:.2f}%")
        print(f"- Différence: {diff:+.2f}% ({diff / paper_accuracy * 100:+.2f}% de la valeur de référence)")

        # Afficher la précision par classe
        print("\nPrécision par classe:")
        for class_name, acc in class_accuracy.items():
            print(f"- {class_name}: {acc * 100:.2f}%")

        # Afficher le rapport détaillé
        formatted_df = report_df.copy()
        for col in ['precision', 'recall', 'f1-score']:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x * 100:.2f}%" if isinstance(x, (int, float)) else x)

        print("\nRapport détaillé:")
        print(formatted_df)

        # NOUVELLE PARTIE: Visualisations améliorées
        # 1. Créer la visualisation combinée (barres + matrice)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Graphique de barres pour la précision globale et par classe
        ax = axes[0]

        # Préparer les données
        class_names = list(class_accuracy.keys())
        class_accuracies = [class_accuracy[cls] * 100 for cls in class_names]

        # Ajouter la précision globale
        all_names = ['Global'] + class_names
        all_accuracies = [accuracy * 100] + class_accuracies

        # Ajouter la référence du papier
        all_names.append('Papier (réf.)')
        all_accuracies.append(paper_accuracy)

        # Créer les barres avec des couleurs distinctes
        bar_colors = ['blue', 'green', 'red', 'purple']
        bars = ax.bar(all_names, all_accuracies, color=bar_colors[:len(all_names)])

        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom')

        # Configurer le graphique
        ax.set_ylim([0, 105])  # Pour avoir de la place pour les étiquettes
        ax.set_ylabel('Précision (%)')
        ax.set_title('Comparaison des précisions')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Matrice de confusion améliorée
        ax = axes[1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=test_dataset.classes,
                    yticklabels=test_dataset.classes,
                    ax=ax)
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Vérité terrain')
        ax.set_title('Matrice de confusion')

        # Finaliser et sauvegarder
        plt.tight_layout()
        plt.savefig('test_evaluation.png', dpi=300)
        print(f"Visualisation complète sauvegardée dans 'test_evaluation.png'")

        # Afficher le graphique si dans un environnement interactif
        try:
            plt.show()
        except:
            pass

        # Retourner un dictionnaire avec tous les résultats
        results = {
            'accuracy': accuracy,
            'accuracy_percent': accuracy * 100,
            'confidence_interval': confidence_interval,
            'class_accuracy': class_accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'classification_report_df': report_df,
            'n_samples': n,
            'class_names': test_dataset.classes
        }

        return results

    except Exception as e:
        print(f"Erreur lors de l'évaluation sur l'ensemble de test: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    chemin_absolu = "C:\\Users\\herve\\OneDrive - Universite de Montreal\\Github\\IFT3710-6759-soccer-analysis\\event detection project\\dataset\\"
    MODEL_PATH = "card_model.pth"
    VAL_DIR = chemin_absolu + "validation"
    TEST_DIR = chemin_absolu + "test"
    PAPER_ACCURACY = 79.90

    # Évaluer le modèle
    #results = evaluate_card_model(MODEL_PATH, VAL_DIR, paper_accuracy=PAPER_ACCURACY)
    test_results = evaluate_on_test_set(MODEL_PATH, TEST_DIR,paper_accuracy=PAPER_ACCURACY)