import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from global_model import GlobalModel, LABELS, LABELS_INDEXES_BY_NAME
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix as cc
import seaborn as sns


def evaluate_on_test_dataset(model, test_dir, output_file="evaluation_results.csv"):
    """
    Évalue le modèle sur l'ensemble de test en utilisant les métriques PyTorch.
    """
    all_image_paths = []
    all_true_labels = []

    # les classes "no-highlight"
    no_highlight_classes = ["Center", "Left", "Right"]

    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name in no_highlight_classes:
            label_idx = LABELS.index("No-highlight") if "No-highlight" in LABELS else -1
        elif class_name == "Cards":
            # Ignorer la classe "Cards" générique
            continue
        else:
            # Pour les autres classes
            label_idx = LABELS.index(class_name) if class_name in LABELS else -1

        if label_idx == -1:
            print(f"Classe {class_name} non trouvée. Ignorer.")
            continue


        # Collecter les chemins d'images et leurs étiquettes
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, img_file)
                all_image_paths.append(img_path)
                all_true_labels.append(label_idx)

    '''
    all_image_paths.append(
        "C:\\Users\\herve\\OneDrive - Universite de Montreal\\Github\\IFT3710-6759-soccer-analysis\\event detection project\\dataset\\test\\Cards\\Cards__1__0.jpg")
    all_image_paths.append(
        "C:\\Users\\herve\\OneDrive - Universite de Montreal\\Github\\IFT3710-6759-soccer-analysis\\event detection project\\dataset\\test\\Cards\\Cards__1__1.jpg")
    all_true_labels.append(LABELS.index('Yellow-Cards'))
    all_true_labels.append(LABELS.index('Yellow-Cards'))
    '''

    # Prédiction avec le modèle
    print(f"Évaluation sur {len(all_image_paths)} images...")

    predicted_labels = model.predict(all_image_paths)

    print("predicted_labels")
    print(predicted_labels)

    true_labels = torch.tensor(all_true_labels)
    pred_labels = torch.tensor(predicted_labels)


    class_results = []
    """"
    for class_idx, class_name in enumerate(LABELS):
        if class_name == "No-highlight":
            continue

        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')  # 'weighted' prend en compte les déséquilibres de classe
        recall = recall_score(true_labels, pred_labels, average='weighted')

        class_results.append({
            'class': class_name,
            'precision': accuracy,
            'recall': recall,
            'f1_score': f1
        })

        print(f"{class_name} - Précision: {accuracy:.4f}, Rappel: {recall:.4f}, F1: {f1:.4f}")
    """
    print('true_labels')
    print(true_labels)
    print('predicted_labels')
    print(pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels,
                  average=None)  # 'weighted' prend en compte les déséquilibres de classe
    recall = recall_score(true_labels, pred_labels, average=None)

    class_results.append({
        'precision': accuracy,
        'recall': recall,
        'f1_score': f1
    })

    cm = cc(true_labels, pred_labels)


    plt.figure(figsize=(6, 5))
    x_labels = list(LABELS_INDEXES_BY_NAME.values())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=x_labels, yticklabels=x_labels)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')
    plt.title('Matrice de Confusion')
    plt.show()

    # Créer un DataFrame et sauvegarder les résultats
    results_df = pd.DataFrame(class_results)
    results_df.to_csv(output_file, index=False)


    print("Matrice de confusion:")
    num_classes = len(LABELS)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(true_labels, pred_labels):
        confusion_matrix[t, p] += 1
    print(confusion_matrix)

    return {
        'class_results': class_results,
        'confusion_matrix': confusion_matrix
    }


if __name__ == "__main__":
    model = GlobalModel(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Évaluer sur l'ensemble de test
    test_dir = "dataset/test"
    results = evaluate_on_test_dataset(model, test_dir)