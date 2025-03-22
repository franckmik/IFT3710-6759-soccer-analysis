import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from card_classifier import CardDetector, ColorEmphasisTransform
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Visualiser la matrice de confusion.

    Args:
        cm (array): Matrice de confusion
        class_names (list): Noms des classes
        save_path (str): Chemin pour sauvegarder l'image
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité terrain')
    plt.title('Matrice de confusion')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Matrice de confusion sauvegardée dans '{save_path}'")


def visualize_dataset_samples(dataset_dir, class_names, num_samples=5, save_path="dataset_samples.png"):
    """
    Visualiser des échantillons du dataset pour chaque classe.

    Args:
        dataset_dir (str): Chemin vers le répertoire du dataset
        class_names (list): Noms des classes
        num_samples (int): Nombre d'échantillons par classe à visualiser
        save_path (str): Chemin pour sauvegarder l'image
    """
    # Transformation minimale qui convertit les images en tensors
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_dir, transform=basic_transform)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Séparer les images par classe
    class_samples = {i: [] for i in range(len(class_names))}

    # Collecter des échantillons pour chaque classe
    for images, labels in loader:
        for img, lbl in zip(images, labels):
            if len(class_samples[lbl.item()]) < num_samples:
                class_samples[lbl.item()].append(img)

        if all(len(samples) >= num_samples for samples in class_samples.values()):
            break

    # Visualiser
    plt.figure(figsize=(num_samples * 3, len(class_names) * 3))

    for class_idx, samples in class_samples.items():
        for i, img in enumerate(samples):
            if i >= num_samples:
                continue

            img_np = img.permute(1, 2, 0).numpy()

            # Appliquer la transformation d'emphase de couleur
            pil_img = transforms.ToPILImage()(img)
            enhanced_img = ColorEmphasisTransform()(pil_img)
            enhanced_np = np.array(enhanced_img) / 255.0

            # Afficher côte à côte original et accentué
            plt.subplot(len(class_names), num_samples * 2, class_idx * num_samples * 2 + i * 2 + 1)
            plt.imshow(img_np)
            plt.title(f"{class_names[class_idx]}\nOriginal")
            plt.axis('off')

            plt.subplot(len(class_names), num_samples * 2, class_idx * num_samples * 2 + i * 2 + 2)
            plt.imshow(enhanced_np)
            plt.title(f"{class_names[class_idx]}\nAccentué")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Échantillons du dataset sauvegardés dans '{save_path}'")


def main():
    parser = argparse.ArgumentParser(description='Entraîner le Détecteur de Couleur de Carton de Football')
    parser.add_argument('data_dir', type=str, help='Chemin vers le répertoire de données d\'entraînement')
    parser.add_argument('--val_dir', type=str, default=None, help='Chemin vers le répertoire de données de validation')
    parser.add_argument('--output', type=str, help='Chemin pour sauvegarder le modèle entraîné',
                        default='card_model.pth')
    parser.add_argument('--epochs', type=int, help='Nombre d\'époques d\'entraînement', default=25)
    parser.add_argument('--batch_size', type=int, help='Taille de batch pour l\'entraînement', default=16)
    parser.add_argument('--lr', type=float, help='Taux d\'apprentissage', default=0.001)
    parser.add_argument('--color_weight', type=float, help='Poids de la perte de couleur', default=0.7)
    parser.add_argument('--val_split', type=float, help='Ratio de validation si pas de val_dir', default=0.2)

    args = parser.parse_args()

    # Définir les transformations avec accent sur la couleur
    train_transform = transforms.Compose([
        ColorEmphasisTransform(saturation_factor=2.0, red_boost=1.6, yellow_boost=1.4),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        ColorEmphasisTransform(saturation_factor=2.0, red_boost=1.6, yellow_boost=1.4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Vérifier l'existence du répertoire de données
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Le répertoire de données {args.data_dir} n'existe pas")

    # Charger les datasets
    print(f"Chargement des données d'entraînement depuis {args.data_dir}")
    try:
        dataset = datasets.ImageFolder(
            args.data_dir,
            transform=None  # Pas de transformation pour visualisation
        )

        # Vérifier que nous avons les classes attendues
        class_names = dataset.classes
        print(f"Classes trouvées: {class_names}")

        # Visualiser quelques échantillons du dataset
        visualize_dataset_samples(args.data_dir, class_names, save_path="dataset_samples.png")

        # Maintenant appliquer les transformations
        train_dataset = datasets.ImageFolder(
            args.data_dir,
            transform=train_transform
        )

        # Si pas de répertoire de validation, diviser le dataset d'entraînement
        if args.val_dir is None:
            val_size = int(len(train_dataset) * args.val_split)
            train_size = len(train_dataset) - val_size

            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            val_dataset.dataset.transform = val_transform

            print(f"Dataset divisé en {train_size} exemples d'entraînement et {val_size} exemples de validation")
        else:
            val_dataset = datasets.ImageFolder(
                args.val_dir,
                transform=val_transform
            )
            print(f"Chargement des données de validation depuis {args.val_dir}")

        # Créer les DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return

    # Initialiser le détecteur
    detector = CardDetector()

    # Entraîner le modèle
    print("Démarrage de l'entraînement...")
    history = detector.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        color_weight=args.color_weight
    )

    # Sauvegarder le modèle entraîné
    detector.save_model(args.output)

    # Évaluer le modèle sur l'ensemble de validation
    detector.model.eval()

    # Calculer les métriques
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(detector.device), labels.to(detector.device)
            outputs, _, _, _, _ = detector.model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculer et afficher la matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png")

    # Afficher le rapport de classification
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nRapport de classification:")
    print(report)

    # Visualiser quelques exemples de prédictions
    detector.visualize_class_predictions(val_loader, num_samples=5, save_path="class_predictions.png")



if __name__ == "__main__":
    main()