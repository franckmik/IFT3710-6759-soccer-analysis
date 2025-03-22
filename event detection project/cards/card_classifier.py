import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ColorEmphasisTransform:
    """
    Transforme l'image pour accentuer la différence entre les couleurs rouge et jaune,
    ce qui devrait aider le modèle à mieux distinguer les cartons.
    """

    def __init__(self, saturation_factor=1.8, red_boost=1.3, yellow_boost=1.3):
        self.saturation_factor = saturation_factor
        self.red_boost = red_boost
        self.yellow_boost = yellow_boost

    def __call__(self, img):
        # Convertir l'image PIL en tableau numpy
        np_img = np.array(img)

        # Convertir de RGB à HSV pour manipuler la saturation et la teinte
        hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Augmenter la saturation pour rendre les couleurs plus vives
        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * self.saturation_factor, 0, 255)

        # Identifier et booster les pixels jaunes (teinte autour de 30)
        yellow_mask = ((hsv_img[:, :, 0] >= 20) & (hsv_img[:, :, 0] <= 35)) & (hsv_img[:, :, 1] > 120)
        hsv_img[:, :, 1][yellow_mask] = np.clip(hsv_img[:, :, 1][yellow_mask] * self.yellow_boost, 0, 255)
        hsv_img[:, :, 2][yellow_mask] = np.clip(hsv_img[:, :, 2][yellow_mask] * self.yellow_boost, 0, 255)

        # Identifier et booster les pixels rouges (teinte proche de 0 ou 180)
        red_mask = ((hsv_img[:, :, 0] <= 8) | (hsv_img[:, :, 0] >= 175)) & (hsv_img[:, :, 1] > 120)
        hsv_img[:, :, 1][red_mask] = np.clip(hsv_img[:, :, 1][red_mask] * self.red_boost, 0, 255)
        hsv_img[:, :, 2][red_mask] = np.clip(hsv_img[:, :, 2][red_mask] * self.red_boost, 0, 255)

        # Reconvertir en RGB
        enhanced_img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Reconvertir en image PIL
        return Image.fromarray(enhanced_img)


class EnhancedColorAttention(nn.Module):
    """
    Module d'attention couleur amélioré qui se concentre spécifiquement sur
    les caractéristiques rouge et jaune pour mieux distinguer les cartons.
    """

    def __init__(self, in_channels):
        super(EnhancedColorAttention, self).__init__()

        # Extraction des caractéristiques de couleur
        self.color_extract = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Attention spécifique pour les caractéristiques rouges
        self.red_attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Attention spécifique pour les caractéristiques jaunes
        self.yellow_attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Transformation pour reconstituer les features d'origine
        self.reconstruct = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x):
        # Extraire les caractéristiques de couleur
        color_features = self.color_extract(x)

        # Appliquer l'attention pour chaque couleur
        red_weights = self.red_attention(color_features)
        yellow_weights = self.yellow_attention(color_features)

        # Combiner les poids d'attention
        combined_weights = self.reconstruct(color_features) * (red_weights + yellow_weights)

        # Rehausser les caractéristiques originales
        enhanced_features = x * combined_weights

        return enhanced_features, red_weights, yellow_weights


class OSME_MAMC_Module(nn.Module):
    """
    One-Squeeze Multi-Excitation (OSME) module avec Multi-Attention Multi-Class constraint (MAMC)
    pour la classification fine-grain des cartons rouges et jaunes.

    Retourner les features d'attention et les cartes pour le calcul de perte et visualisation
    """

    def __init__(self, in_channels, attention_num=2):
        super(OSME_MAMC_Module, self).__init__()
        self.attention_num = attention_num

        # Pooling global
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Modules d'attention
        self.attention_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(attention_num)
        ])

        # Couches fully connected pour chaque attention
        self.fcs = nn.ModuleList([
            nn.Linear(in_channels, 512) for _ in range(attention_num)
        ])

        # Classifier final combinant toutes les attentions
        self.classifier = nn.Linear(512 * attention_num, 2)


    def forward(self, x):
        batch_size = x.size(0)
        attention_outputs = []
        attention_maps = []

        for i in range(self.attention_num):
            # Appliquer l'attention
            att = self.attention_modules[i](x)
            attention_maps.append(att)
            attended_feat = x * att

            # Pooling global et reshape
            pooled = self.global_pool(attended_feat).view(batch_size, -1)

            # Appliquer la couche FC
            fc_out = self.fcs[i](pooled)
            attention_outputs.append(fc_out)

        # Concaténer toutes les sorties d'attention
        combined = torch.cat(attention_outputs, dim=1)

        output = self.classifier(combined)


        return output, attention_outputs, attention_maps


class ColorAwareLoss(nn.Module):
    """
    Fonction de perte personnalisée qui prend en compte les caractéristiques de couleur
    pour mieux distinguer les cartons rouges et jaunes.

    Classe 0 = carton jaune
    Classe 1 = carton rouge
    """

    def __init__(self, color_weight=0.5):
        super(ColorAwareLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.color_weight = color_weight

    def forward(self, outputs, targets, red_attention, yellow_attention):
        # Perte de classification standard
        ce_loss = self.ce_loss(outputs, targets)

        # Perte d'attention couleur
        batch_size = targets.size(0)
        color_loss = torch.tensor(0.0, device=outputs.device)

        # Encourager l'attention d'une couleur, décourager l'attention de l'autre
        for i in range(batch_size):
            if targets[i] == 0:
                color_loss += (1 - yellow_attention[i].mean()) + red_attention[i].mean()
            else:

                color_loss += (1 - red_attention[i].mean()) + yellow_attention[i].mean()

        color_loss /= batch_size

        # Perte combinée
        total_loss = ce_loss + self.color_weight * color_loss

        return total_loss


class CardClassifier(nn.Module):
    """
    Classificateur amélioré pour distinguer entre cartons rouges et jaunes.
    Utilise EfficientNetB0 avec attention couleur renforcée et module OSME+MAMC.
    """

    def __init__(self):
        super(CardClassifier, self).__init__()

        # Charger le modèle EfficientNetB0 (pré-entraîné)
        self.base_model = models.efficientnet_b0(pretrained=True)

        # Extraire l'extracteur de features
        self.features = self.base_model.features

        # Remplacer la couche d'attention de couleur par notre version améliorée
        self.color_attention = EnhancedColorAttention(in_channels=1280)

        # Ajouter le module OSME+MAMC
        self.osme_mamc = OSME_MAMC_Module(in_channels=1280)  # EfficientNetB0 produit 1280 canaux

        # Pour Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Extraire les features de base
        features = self.features(x)

        # Enregistrer les activations pour Grad-CAM
        self.activations = features

        # Enregistrer le hook pour le gradient
        if not self.training and torch.is_grad_enabled():
            try:
                h = features.register_hook(self.activations_hook)
            except RuntimeError:
                pass

        # Appliquer l'attention sur la couleur
        enhanced_features, red_attention, yellow_attention = self.color_attention(features)

        # Appliquer le module OSME+MAMC
        outputs, attention_features, attention_maps = self.osme_mamc(enhanced_features)

        # Retourner les attentions de couleur pour le calcul de perte
        return outputs, attention_features, attention_maps, red_attention, yellow_attention

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations


class GradCAM:
    """
    Implémentation de Grad-CAM pour visualiser les parties de l'image sur lesquelles
    le modèle se concentre pour sa classification.
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def generate_cam(self, input_image, target_class=None):
        # Mettre le modèle en mode évaluation
        self.model.eval()

        # Assurer que le calcul des gradients est activé
        with torch.enable_grad():
            # Réinitialiser les gradients
            self.model.zero_grad()

            # Faire passer l'image dans le modèle
            outputs, _, _, _, _ = self.model(input_image)

            # Si pas de classe cible spécifiée, prendre la classe avec le plus haut score
            if target_class is None:
                target_class = outputs.argmax(dim=1).item()

            # Calculer les gradients par rapport à la classe cible
            target = outputs[0, target_class]
            target.backward()

            # Obtenir les gradients et les activations
            gradients = self.model.get_activations_gradient()
            activations = self.model.get_activations()

            if gradients is None or activations is None:
                print("Avertissement: gradients ou activations indisponibles")
                return np.zeros((224, 224))  # Retourner une carte vide

            # Pooler les gradients sur la dimension spatiale
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            # Pondérer les canaux d'activation par les gradients
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]

            # Moyenne sur les canaux et appliquer ReLU
            heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu()
            heatmap = np.maximum(heatmap, 0)

            # Normaliser la heatmap
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_img, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # Redimensionner la heatmap à la taille de l'image originale
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        # Appliquer une colormap
        heatmap = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap, colormap)

        # Convertir l'image originale en BGR si nécessaire
        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
            original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        else:
            original_img_bgr = np.repeat(original_img[:, :, np.newaxis], 3, axis=2)

        # Superposer la heatmap sur l'image originale
        overlay = cv2.addWeighted(original_img_bgr, 1 - alpha, colored_heatmap, alpha, 0)

        # Reconvertir en RGB pour affichage avec matplotlib
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


class CardDetector:
    """
    Détecteur de carton amélioré avec une meilleure attention aux couleurs.
    """

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CardClassifier().to(self.device)

        # Définir les transformations d'image avec emphase sur la couleur
        self.transform = transforms.Compose([
            ColorEmphasisTransform(saturation_factor=2.0, red_boost=1.6, yellow_boost=1.4),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Charger les poids pré-entraînés si fournis
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Modèle chargé avec succès depuis {model_path}")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")

        self.model.eval()
        self.classes = ['Carton Rouge','Carton Jaune']

        # Initialiser Grad-CAM
        self.grad_cam = GradCAM(self.model)

    def predict(self, image_path, visualize_attention=False):
        """
        Prédire la couleur du carton (rouge ou jaune) pour une image donnée.

        Args:
            image_path (str): Chemin vers l'image d'entrée
            visualize_attention (bool): Si True, génère et affiche la visualisation Grad-CAM

        Returns:
            str: Couleur de carton prédite ('Carton Jaune' or 'Carton Rouge')
            float: Score de confiance (0-1)
            np.ndarray (optionnel): Visualisation Grad-CAM si visualize_attention=True
        """
        # Ouvrir l'image
        original_image = Image.open(image_path).convert('RGB')
        original_np = np.array(original_image)

        # Appliquer les transformations
        image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)

        # Obtenir la prédiction
        with torch.no_grad():
            outputs, _, _, _, _ = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        if visualize_attention:
            # Générer la carte de chaleur Grad-CAM
            cam = self.grad_cam.generate_cam(image_tensor, predicted.item())

            # Superposer sur l'image originale
            visualization = self.grad_cam.overlay_heatmap(cam, original_np)

            return self.classes[predicted.item()], confidence.item(), visualization

        return self.classes[predicted.item()], confidence.item()

    def train(self, train_loader, val_loader=None, num_epochs=20, learning_rate=0.001, color_weight=0.5):
        """
        Entraîner le modèle de classificateur de carton.

        Args:
            train_loader (DataLoader): DataLoader pour les données d'entraînement
            val_loader (DataLoader, optional): DataLoader pour les données de validation
            num_epochs (int): Nombre d'époques d'entraînement
            learning_rate (float): Taux d'apprentissage pour l'optimiseur
            color_weight (float): Poids de la perte de couleur dans la fonction de perte totale
        """
        # Mettre le modèle en mode entraînement
        self.model.train()

        # Définir la fonction de perte et l'optimiseur
        color_loss = ColorAwareLoss(color_weight=color_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Historique d'entraînement
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        # Boucle d'entraînement
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Réinitialiser les gradients
                optimizer.zero_grad()

                # Forward pass
                outputs, attention_features, _, red_attention, yellow_attention = self.model(inputs)

                # Calcul de la perte avec attention couleur
                loss = color_loss(outputs, labels, red_attention, yellow_attention)

                # Backward pass et optimisation
                loss.backward()
                optimizer.step()

                # Suivre les statistiques
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Valider si un validation loader est fourni
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs, attention_features, _, red_attention, yellow_attention = self.model(inputs)
                        loss = color_loss(outputs, labels, red_attention, yellow_attention)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_loss = val_loss / len(val_loader)
                val_acc = 100 * val_correct / val_total

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Ajuster le learning rate
                scheduler.step(val_loss)

                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

                self.model.train()
            else:
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

        # Tracer l'historique d'entraînement
        self._plot_training_history(history, num_epochs)

        return history

    def _plot_training_history(self, history, num_epochs):
        """
        Visualiser l'historique d'entraînement.

        Args:
            history (dict): Dictionnaire contenant les métriques d'entraînement et de validation
            num_epochs (int): Nombre d'époques d'entraînement
        """
        plt.figure(figsize=(12, 5))

        # Tracer la perte
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(range(1, num_epochs + 1), history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Tracer la précision
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), history['train_acc'], label='Train Acc')
        if 'val_acc' in history and history['val_acc']:
            plt.plot(range(1, num_epochs + 1), history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

        print("Historique d'entraînement sauvegardé dans 'training_history.png'")

    def visualize_class_predictions(self, data_loader, num_samples=5, save_path="class_predictions.png"):
        """
        Visualiser les prédictions du modèle sur quelques échantillons du dataset.

        Args:
            data_loader (DataLoader): DataLoader contenant les données
            num_samples (int): Nombre d'échantillons à visualiser
            save_path (str): Chemin pour sauvegarder la visualisation
        """
        self.model.eval()

        # Obtenir quelques échantillons
        images, labels = [], []
        class_counts = {0: 0, 1: 0}

        for batch_images, batch_labels in data_loader:
            for i in range(len(batch_labels)):
                label = batch_labels[i].item()
                if label in class_counts and class_counts[label] < num_samples:
                    images.append(batch_images[i])
                    labels.append(batch_labels[i])
                    class_counts[label] += 1

            if all(count >= num_samples for count in class_counts.values()):
                break

        # Convertir en tensors
        if not images:
            print("Aucun exemple trouvé pour la visualisation")
            return

        images = torch.stack(images).to(self.device)
        labels = torch.stack(labels).to(self.device)

        # Obtenir les prédictions
        with torch.no_grad():
            outputs, _, _, red_attention, yellow_attention = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(probabilities, 1)

        # Visualiser
        plt.figure(figsize=(15, num_samples * 3))
        for i in range(len(images)):
            # Convertir l'image en numpy array
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)

            # Générer la heatmap Grad-CAM
            with torch.set_grad_enabled(True):
                cam = self.grad_cam.generate_cam(images[i].unsqueeze(0), predictions[i].item())

            cam_img = self.grad_cam.overlay_heatmap(cam, (img * 255).astype(np.uint8))

            # Afficher l'image originale
            plt.subplot(len(images), 2, 2 * i + 1)
            plt.imshow(img)
            plt.title(f"Vraie: {self.classes[labels[i]]}\nPrédiction: {self.classes[predictions[i]]}")
            plt.axis('off')

            # Afficher la visualisation Grad-CAM
            plt.subplot(len(images), 2, 2 * i + 2)
            plt.imshow(cam_img)
            plt.title("Grad-CAM")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Visualisation des prédictions sauvegardée dans '{save_path}'")

    def save_model(self, path):
        """Sauvegarder le modèle entraîné dans un fichier."""
        torch.save(self.model.state_dict(), path)
        print(f"Modèle sauvegardé dans {path}")