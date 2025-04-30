import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from card_classifier import CardDetector
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):

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

    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(dataset_dir, transform=basic_transform)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    class_samples = {i: [] for i in range(len(class_names))}

    for images, labels in loader:
        for img, lbl in zip(images, labels):
            if len(class_samples[lbl.item()]) < num_samples:
                class_samples[lbl.item()].append(img)

        if all(len(samples) >= num_samples for samples in class_samples.values()):
            break

    plt.figure(figsize=(num_samples * 3, len(class_names) * 3))

    for class_idx, samples in class_samples.items():
        for i, img in enumerate(samples):
            if i >= num_samples:
                continue

            plt.subplot(len(class_names), num_samples, class_idx * num_samples + i + 1)
            plt.imshow(img.permute(1, 2, 0).numpy())
            plt.title(f"{class_names[class_idx]}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Échantillons du dataset sauvegardés dans '{save_path}'")


def main():

    from absolute_path import get_file_path

    parser = argparse.ArgumentParser(description='Entraîner le Détecteur de Couleur de Carton de Football')
    parser.add_argument('--data_dir', type=str,
                        default= str(get_file_path("event detection project\\dataset\\train")),
                        help='Chemin vers le répertoire de données d\'entraînement')
    parser.add_argument('--val_dir', type=str,
                        default=str(get_file_path("event detection project\\dataset\\validation")),
                        help='Chemin vers le répertoire de données de validation')
    parser.add_argument('--output', type=str,
                        default=str(get_file_path("cards\\card_model.pth")),
                        help='Chemin pour sauvegarder le modèle entraîné')
    parser.add_argument('--epochs', type=int,
                        default=60,
                        help='Nombre maximum d\'époques d\'entraînement')
    parser.add_argument('--batch_size', type=int,
                        default=16,
                        help='Taille de batch pour l\'entraînement')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='Taux d\'apprentissage')
    parser.add_argument('--patience', type=int,
                        default=5,
                        help='Patience pour early stopping')
    parser.add_argument('--val_split', type=float,
                        default=0.2,
                        help='Ratio de validation si pas de val_dir')


    args = parser.parse_args()


    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Le répertoire de données {args.data_dir} n'existe pas")


    print(f"Chargement des données d'entraînement depuis {args.data_dir}")
    try:
        dataset = datasets.ImageFolder(
            args.data_dir,
            transform=None
        )

        class_names = dataset.classes
        print(f"Classes trouvées: {class_names}")

        visualize_dataset_samples(args.data_dir, class_names, save_path="dataset_samples.png")

        train_dataset = datasets.ImageFolder(
            args.data_dir,
            transform=train_transform
        )


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

    detector = CardDetector()

    print("Démarrage de l'entraînement...")
    history = detector.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        batch_size=args.batch_size
    )


    detector.save_model(args.output)

    detector.model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(detector.device), labels.to(detector.device)
            outputs, _, _, _, _ = detector.model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png")

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nRapport de classification:")
    print(report)

if __name__ == "__main__":
    main()