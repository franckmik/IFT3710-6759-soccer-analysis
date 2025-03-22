import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from global_model import GlobalModel, LABELS


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

    # Prédiction avec le modèle
    print(f"Évaluation sur {len(all_image_paths)} images...")
    predicted_labels = model.predict(all_image_paths)

    true_labels = torch.tensor(all_true_labels)
    pred_labels = torch.tensor(predicted_labels)

    class_results = []
    for class_idx, class_name in enumerate(LABELS):
        if class_name == "No-highlight":
            continue

        # Calculer la précision pour cette classe (true positives / predicted positives)
        true_positives = ((pred_labels == class_idx) & (true_labels == class_idx)).sum().item()
        predicted_positives = (pred_labels == class_idx).sum().item()

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0

        # Calculer le rappel pour cette classe (true positives / actual positives)
        actual_positives = (true_labels == class_idx).sum().item()
        recall = true_positives / actual_positives if actual_positives > 0 else 0

        # Calculer le F1-score (2 * precision * recall / (precision + recall))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_results.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        print(f"{class_name} - Précision: {precision:.4f}, Rappel: {recall:.4f}, F1: {f1:.4f}")

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
    model = GlobalModel()

    # Évaluer sur l'ensemble de test
    test_dir = "dataset/test"
    results = evaluate_on_test_dataset(model, test_dir)