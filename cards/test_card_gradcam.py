import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models


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

        # Classification finale
        output = self.classifier(combined)

        return output, attention_outputs, attention_maps


class CardClassifier(nn.Module):
    def __init__(self):
        super(CardClassifier, self).__init__()

        # Charger le modèle EfficientNetB0 (pré-entraîné)
        self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Extraire l'extracteur de features
        self.features = self.base_model.features

        # Remplacer le module d'attention de couleur simple par l'amélioré
        self.color_attention = EnhancedColorAttention(in_channels=1280)

        # Ajouter le module OSME+MAMC
        self.osme_mamc = OSME_MAMC_Module(in_channels=1280)

        # Pour Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Extraire les features de base
        features = self.features(x)

        # Stocker les activations pour Grad-CAM
        self.activations = features

        # Enregistrer le hook seulement si nécessaire
        if not self.training and torch.is_grad_enabled():
            try:
                h = features.register_hook(self.activations_hook)
            except RuntimeError:
                pass

        # Appliquer l'attention sur la couleur
        enhanced_features, red_attention, yellow_attention = self.color_attention(features)

        # Appliquer le module OSME+MAMC
        outputs, attention_features, attention_maps = self.osme_mamc(enhanced_features)

        return outputs, attention_features, attention_maps, red_attention, yellow_attention

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations


class GradCAM:
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


class ColorEmphasisTransform:
    """
    Transforme l'image pour accentuer la différence entre les couleurs rouge et jaune.
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
        yellow_mask = ((hsv_img[:, :, 0] >= 20) & (hsv_img[:, :, 0] <= 40)) & (hsv_img[:, :, 1] > 100)
        hsv_img[:, :, 1][yellow_mask] = np.clip(hsv_img[:, :, 1][yellow_mask] * self.yellow_boost, 0, 255)
        hsv_img[:, :, 2][yellow_mask] = np.clip(hsv_img[:, :, 2][yellow_mask] * self.yellow_boost, 0, 255)

        # Identifier et booster les pixels rouges (teinte proche de 0 ou 180)
        red_mask = ((hsv_img[:, :, 0] <= 10) | (hsv_img[:, :, 0] >= 170)) & (hsv_img[:, :, 1] > 100)
        hsv_img[:, :, 1][red_mask] = np.clip(hsv_img[:, :, 1][red_mask] * self.red_boost, 0, 255)
        hsv_img[:, :, 2][red_mask] = np.clip(hsv_img[:, :, 2][red_mask] * self.red_boost, 0, 255)

        # Reconvertir en RGB
        enhanced_img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Reconvertir en image PIL
        return Image.fromarray(enhanced_img)


class CardDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CardClassifier().to(self.device)

        # Définir les transformations d'image avec emphase sur la couleur
        self.transform = transforms.Compose([
            ColorEmphasisTransform(saturation_factor=1.8, red_boost=1.3, yellow_boost=1.3),
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
        self.classes = ['Carton Rouge', 'Carton Jaune']

        # Initialiser Grad-CAM
        self.grad_cam = GradCAM(self.model)

    def predict(self, image_path, visualize_attention=False):
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


def main():
    parser = argparse.ArgumentParser(description='Détecteur de Couleur de Carton de Football')
    parser.add_argument('image_path', type=str, help='Chemin vers l\'image d\'entrée')
    parser.add_argument('--model', type=str, help='Chemin vers le modèle entraîné', default=None)
    parser.add_argument('--visualize', action='store_true', help='Visualiser le résultat de détection avec Grad-CAM')

    args = parser.parse_args()

    # Initialiser le détecteur
    detector = CardDetector(model_path=args.model)

    # Prédire la couleur du carton
    if args.visualize:
        prediction, confidence, visualization = detector.predict(args.image_path, visualize_attention=True)

        # Afficher le résultat
        plt.figure(figsize=(12, 6))

        # Image originale
        original_img = cv2.imread(args.image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f"Original - {prediction} ({confidence:.2f})")
        plt.axis('off')

        # Visualisation Grad-CAM
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title("Zones d'attention (Grad-CAM)")
        plt.axis('off')

        plt.tight_layout()
        output_path = f"gradcam_{args.image_path.split('/')[-1].split('\\')[-1]}"
        plt.savefig(output_path)
        plt.show()

        print(f"Détecté: {prediction} avec une confiance de {confidence:.2f}")
        print(f"Visualisation sauvegardée dans {output_path}")
    else:
        prediction, confidence = detector.predict(args.image_path)
        print(f"Détecté: {prediction} avec une confiance de {confidence:.2f}")


if __name__ == "__main__":
    main()