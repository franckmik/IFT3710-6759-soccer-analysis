import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ColorEmphasisTransform:

    def __init__(self, saturation_factor=1.8, red_boost=1.3, yellow_boost=1.3):
        self.saturation_factor = saturation_factor
        self.red_boost = red_boost
        self.yellow_boost = yellow_boost

    def __call__(self, img):
        np_img = np.array(img)

        hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV).astype(np.float32)

        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * self.saturation_factor, 0, 255)

        yellow_mask = ((hsv_img[:, :, 0] >= 20) & (hsv_img[:, :, 0] <= 35)) & (hsv_img[:, :, 1] > 120)
        hsv_img[:, :, 1][yellow_mask] = np.clip(hsv_img[:, :, 1][yellow_mask] * self.yellow_boost, 0, 255)
        hsv_img[:, :, 2][yellow_mask] = np.clip(hsv_img[:, :, 2][yellow_mask] * self.yellow_boost, 0, 255)

        red_mask = ((hsv_img[:, :, 0] <= 8) | (hsv_img[:, :, 0] >= 175)) & (hsv_img[:, :, 1] > 120)
        hsv_img[:, :, 1][red_mask] = np.clip(hsv_img[:, :, 1][red_mask] * self.red_boost, 0, 255)
        hsv_img[:, :, 2][red_mask] = np.clip(hsv_img[:, :, 2][red_mask] * self.red_boost, 0, 255)

        enhanced_img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return Image.fromarray(enhanced_img)

    """
    Note : 
    Le Module d'attention couleur amélioré qui se concentre spécifiquement sur
    les caractéristiques rouge et jaune pour mieux distinguer les cartons a ete desactivé car il n'apporté aucune amelioration
    """
class EnhancedColorAttention(nn.Module):


    def __init__(self, in_channels):
        super(EnhancedColorAttention, self).__init__()

        self.color_extract = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.red_attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.yellow_attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.reconstruct = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x):
        color_features = self.color_extract(x)

        red_weights = self.red_attention(color_features)
        yellow_weights = self.yellow_attention(color_features)

        combined_weights = self.reconstruct(color_features) * (red_weights + yellow_weights)

        enhanced_features = x * combined_weights

        return enhanced_features, red_weights, yellow_weights


class OSME_MAMC_Module(nn.Module):
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
            att = self.attention_modules[i](x)
            attention_maps.append(att)
            attended_feat = x * att

            pooled = self.global_pool(attended_feat).view(batch_size, -1)

            fc_out = self.fcs[i](pooled)
            attention_outputs.append(fc_out)

        combined = torch.cat(attention_outputs, dim=1)

        output = self.classifier(combined)

        return output, attention_outputs, attention_maps


class CardClassifier(nn.Module):
    def __init__(self):
        super(CardClassifier, self).__init__()

        # Charger le modèle EfficientNetB0 (pré-entraîné)
        self.base_model = models.efficientnet_b0(pretrained=True)

        # Extraire l'extracteur de features
        self.features = self.base_model.features

        # Remplacer la couche d'attention de couleur par notre version améliorée
        self.color_attention = EnhancedColorAttention(in_channels=1280)

        # Ajouter le module OSME+MAMC
        # EfficientNetB0 produit 1280 canaux
        self.osme_mamc = OSME_MAMC_Module(in_channels=1280)

        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    """
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
    """

    def forward(self, x):
        features = self.features(x)

        self.activations = features

        if not self.training and torch.is_grad_enabled():
            try:
                h = features.register_hook(self.activations_hook)
            except RuntimeError:
                pass

        # SKIP color attention
        # enhanced_features, red_attention, yellow_attention = self.color_attention(features)

        # Creer dummy attention maps (same shape as the original ones)
        dummy_red_attention = torch.ones_like(features[:, :1, :, :])
        dummy_yellow_attention = torch.ones_like(features[:, :1, :, :])

        # Appliquer le module OSME+MAMC directement sur les features originales
        outputs, attention_features, attention_maps = self.osme_mamc(features)

        return outputs, attention_features, attention_maps, dummy_red_attention, dummy_yellow_attention

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
        self.model.eval()

        with torch.enable_grad():
            self.model.zero_grad()

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
                return np.zeros((224, 224))

            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

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
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

        heatmap = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap, colormap)

        # Convertir l'image originale en BGR si nécessaire
        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
            original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        else:
            original_img_bgr = np.repeat(original_img[:, :, np.newaxis], 3, axis=2)

        overlay = cv2.addWeighted(original_img_bgr, 1 - alpha, colored_heatmap, alpha, 0)

        # Reconvertir en RGB pour affichage avec matplotlib
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


class CardDetector:

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CardClassifier().to(self.device)

        # Définir les transformations d'image conformes au tableau V
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Modèle chargé avec succès depuis {model_path}")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")

        self.model.eval()
        self.classes = ['Carton Rouge', 'Carton Jaune']

        self.grad_cam = GradCAM(self.model)

    def predict(self, image_path, visualize_attention=False):

        original_image = Image.open(image_path).convert('RGB')
        original_np = np.array(original_image)

        image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, _, _, _, _ = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        if visualize_attention:
            cam = self.grad_cam.generate_cam(image_tensor, predicted.item())

            visualization = self.grad_cam.overlay_heatmap(cam, original_np)

            return self.classes[predicted.item()], confidence.item(), visualization

        return self.classes[predicted.item()], confidence.item()

    def train(self, train_loader, val_loader=None, num_epochs=60, learning_rate=0.001, patience=5, batch_size=16):
        """
        Entraîner le modèle de classificateur de carton avec les paramètres du tableau V.

        Args:
            train_loader (DataLoader): DataLoader pour les données d'entraînement
            val_loader (DataLoader, optional): DataLoader pour les données de validation
            num_epochs (int): Nombre maximum d'époques d'entraînement
            learning_rate (float): Taux d'apprentissage pour l'optimiseur
            patience (int): Nombre d'époques à attendre sans amélioration avant d'arrêter l'entraînement
            batch_size (int): Taille du batch (16 selon tableau V)
        """
        if train_loader.batch_size != batch_size:
            print(f"Attention: La taille du batch dans le train_loader ({train_loader.batch_size}) "
                  f"ne correspond pas à la taille spécifiée ({batch_size})")

        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Historique d'entraînement
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        # Variables pour early stopping
        best_val_loss = float('inf')
        no_improve_epochs = 0
        best_model_state = None

        # Boucle d'entraînement
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs, _, _, _, _ = self.model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs, _, _, _, _ = self.model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_loss = val_loss / len(val_loader)
                val_acc = 100 * val_correct / val_total

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                    best_model_state = self.model.state_dict().copy()
                    print(f"Nouveau meilleur modèle sauvegardé avec perte de validation: {val_loss:.4f}")
                else:
                    no_improve_epochs += 1
                    print(f"Pas d'amélioration depuis {no_improve_epochs} époques")
                    if no_improve_epochs >= patience and False:
                        print(f"Early stopping déclenché après {epoch + 1} époques")
                        self.model.load_state_dict(best_model_state)
                        break

                self.model.train()
            else:
                print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')


        if val_loader and best_model_state and no_improve_epochs < patience:
            self.model.load_state_dict(best_model_state)
            print("Entraînement terminé. Chargement du meilleur modèle.")

        self._plot_training_history(history, epoch + 1)

        return history

    def _plot_training_history(self, history, num_epochs):
        """
        Visualiser l'historique d'entraînement avec un style scientifique moderne.
        """
        plt.figure(figsize=(15, 6))

        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except (ImportError, ModuleNotFoundError):
            plt.style.use('default')
            plt.rc('axes', grid=True)
            plt.rc('grid', linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 1)
        epochs = range(1, num_epochs + 1)

        plt.plot(epochs, history['train_loss'], 'ro-', linewidth=1.5, markersize=4, label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(epochs, history['val_loss'], 'bo-', linewidth=1.5, markersize=4, label='Val Loss')


        plt.yscale('log')

        plt.grid(True, alpha=0.3)
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        train_acc = history['train_acc']

        train_acc_scaled = [acc / 100 for acc in train_acc]

        plt.plot(epochs, train_acc_scaled, 'ro-', linewidth=1.5, markersize=4, label='Train Acc')
        if 'val_acc' in history and history['val_acc']:
            val_acc_scaled = [acc / 100 for acc in history['val_acc']]
            plt.plot(epochs, val_acc_scaled, 'bo-', linewidth=1.5, markersize=4, label='Val Acc')


        try:
            plt.yscale('symlog', linthresh=0.5)
        except ValueError:
            plt.yscale('log')
            plt.ylim([0.2, 1.05])

        plt.ylim([0.2, 1.05])

        for acc in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
            plt.axhline(y=acc, color='gray', alpha=0.3, linestyle='--')

        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        plt.close()

        print("Historique d'entraînement sauvegardé dans 'training_history.png' et 'training_history.pdf'")


        self._save_run_history('CardClassifier', history, num_epochs)

    def _save_run_history(self, config_name, history, num_epochs, all_results_file='all_training_results.pkl'):
        """Sauvegarde l'historique d'entraînement pour une analyse comparative future"""
        import os
        import pickle

        formatted_history = {
            'accuracy': [acc / 100 for acc in history['train_acc']],
            'loss': history['train_loss'],
            'val_accuracy': [acc / 100 for acc in history['val_acc']] if 'val_acc' in history and history[
                'val_acc'] else [],
            'val_loss': history['val_loss'] if 'val_loss' in history and history['val_loss'] else [],
            'epochs': num_epochs
        }

        all_results = {}
        if os.path.exists(all_results_file):
            with open(all_results_file, 'rb') as f:
                try:
                    all_results = pickle.load(f)
                except Exception as e:
                    print(f"Erreur lors du chargement des résultats précédents: {e}")

        all_results[config_name] = formatted_history

        try:
            with open(all_results_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"Résultats pour '{config_name}' sauvegardés dans '{all_results_file}'")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des résultats: {e}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Modèle sauvegardé dans {path}")
