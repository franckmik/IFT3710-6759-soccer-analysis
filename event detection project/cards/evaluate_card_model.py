import torch
import torch.nn as nn
from fontTools.misc.classifyTools import Classifier
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

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = val_dataset.classes
    print(f"Classes: {class_names}")

    detector.model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(detector.device), labels.to(detector.device)
            outputs, _, _, _, _ = detector.model(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        mask = (all_labels == i)
        if np.sum(mask) > 0:
            class_accuracy[class_name] = np.mean(all_preds[mask] == all_labels[mask])
        else:
            class_accuracy[class_name] = 0.0

    # Obtenir un rapport de classification détaillé
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    report_df = pd.DataFrame(report).transpose()

    n = len(all_labels)
    confidence_interval = 1.96 * np.sqrt((accuracy * (1 - accuracy)) / n)

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
    Affiche les résultats d'évaluation

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

    print("\nPrécision par classe:")
    for class_name, acc in results['class_accuracy'].items():
        print(f"- {class_name}: {acc * 100:.2f}%")

    print("\nMétriques détaillées par classe:")
    report_df = results['classification_report_df']

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


    ax = axes[0]

    class_names = list(results['class_accuracy'].keys())
    class_accuracies = [results['class_accuracy'][cls] * 100 for cls in class_names]

    # Ajouter la précision globale
    all_names = ['Global'] + class_names
    all_accuracies = [results['accuracy_percent']] + class_accuracies

    # Ajouter la référence du papier
    all_names.append('Papier (réf.)')
    all_accuracies.append(paper_accuracy)

    bars = ax.bar(all_names, all_accuracies, color=['blue', 'green', 'red', 'purple'])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom')

    ax.set_ylim([0, 105])
    ax.set_ylabel('Précision (%)')
    ax.set_title('Comparaison des précisions')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax = axes[1]
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=results['class_names'],
                yticklabels=results['class_names'],
                ax=ax)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')
    ax.set_title('Matrice de confusion')

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

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(test_dir):
        print(f"ERREUR: Le répertoire de test '{test_dir}' n'existe pas!")
        return None


    try:
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f"Ensemble de test chargé: {len(test_dataset)} images")
        print(f"Classes: {test_dataset.classes}")

        # Passer en mode évaluation
        detector.model.eval()

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


        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)


        class_accuracy = {}
        for i, class_name in enumerate(test_dataset.classes):
            mask = (all_labels == i)
            if np.sum(mask) > 0:
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


        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        ax = axes[0]

        class_names = list(class_accuracy.keys())
        class_accuracies = [class_accuracy[cls] * 100 for cls in class_names]

        all_names = ['Global'] + class_names
        all_accuracies = [accuracy * 100] + class_accuracies

        all_names.append('Papier (réf.)')
        all_accuracies.append(paper_accuracy)

        bar_colors = ['blue', 'green', 'red', 'purple']
        bars = ax.bar(all_names, all_accuracies, color=bar_colors[:len(all_names)])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom')

        ax.set_ylim([0, 105])
        ax.set_ylabel('Précision (%)')
        ax.set_title('Comparaison des précisions')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax = axes[1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=test_dataset.classes,
                    yticklabels=test_dataset.classes,
                    ax=ax)
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Vérité terrain')
        ax.set_title('Matrice de confusion')

        plt.tight_layout()
        plt.savefig('test_evaluation.png', dpi=300)
        print(f"Visualisation complète sauvegardée dans 'test_evaluation.png'")

        try:
            plt.show()
        except:
            pass

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
    from absolute_path import get_file_path

    MODEL_PATH = str(get_file_path("cards\\card_model.pth"))
    VAL_DIR = str(get_file_path("event detection project\\dataset\\validation"))
    TEST_DIR = str(get_file_path("event detection project\\dataset\\test"))
    PAPER_ACCURACY = 79.90

    # Évaluer le modèle
    results = evaluate_card_model(MODEL_PATH, VAL_DIR, paper_accuracy=PAPER_ACCURACY)
    test_results = evaluate_on_test_set(MODEL_PATH, TEST_DIR,paper_accuracy=PAPER_ACCURACY)