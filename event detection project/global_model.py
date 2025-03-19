import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_model.vae import VAE, vae_loss, recon_loss
from cards.card_classifier import CardClassifier, CardDetector
from my_module import SoccerEventClassifier
from PIL import Image
from torchvision import transforms

VAE_PAPER_THRESHOLD = 328

LABELS = [
    'Free-Kick',
    'To-Subtitue',
    'Corner',
    'Penalty',
    'Red-Cards',
    'Tackle',
    'Yellow-Cards',
    'No-highlight'
]

LABELS_INDEXES_BY_NAME = {key: i for i, key in enumerate(LABELS)}

class GlobalModel:
    def __init__(self, device='cpu'):
        self.device = device

        # Charger les modèles
        self.vae_model = VAE().to(device)
        self.vae_model.load_state_dict(torch.load('vae_model/vae_model_upsample_4.pth', map_location=device))
        self.vae_model.eval()

        self.card_model = CardClassifier().to(device)
        self.card_model.load_state_dict(torch.load('cards/card_model_color_enhanced.pth', map_location=device))
        self.card_model.eval()

        self.image_classifier_model = SoccerEventClassifier().to(device)
        self.image_classifier_model.load_state_dict(torch.load('model_event_classifier.pth', map_location=device))
        self.image_classifier_model.eval()

        # Transformations des images
        self.vae_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.classifier_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
        ])

    def predict(self, image_paths):
        """
        Prend une liste de chemins d'images et retourne leurs prédictions finales.
        """
        predictions = []

        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')

            # === Étape 1: Vérification avec le VAE (No-Highlight) ===
            x = self.vae_transform(image).unsqueeze(0).to(self.device)  # Ajouter batch dimension
            recon_x, mu, logvar = self.vae_model(x)
            #loss = vae_loss(recon_x, x, mu, logvar)
            loss = recon_loss(recon_x, x)

            #print("loss")
            #print(loss)

            if loss >= VAE_PAPER_THRESHOLD:
                predictions.append(LABELS_INDEXES_BY_NAME["No-highlight"])
                continue

            # === Étape 2: Classification d'événement ===
            x = self.classifier_transform(image).unsqueeze(0).to(self.device)
            image_classifier_outputs = self.image_classifier_model(x)
            _, predicted_index = torch.max(image_classifier_outputs.data, 1)
            predicted_index = predicted_index.item()  # Conversion en int

            #print("predicted_index")
            #print(predicted_index)

            if predicted_index in [self.image_classifier_model.class_names_dict["Left"],
                                   self.image_classifier_model.class_names_dict["Right"],
                                   self.image_classifier_model.class_names_dict["Center"]]:
                predictions.append(LABELS_INDEXES_BY_NAME["No-highlight"])
                continue

            if predicted_index != self.image_classifier_model.class_names_dict["Cards"]:
                predictions.append(LABELS_INDEXES_BY_NAME[self.image_classifier_model.class_names[predicted_index]])
                continue

            # === Étape 3: Classification des cartes (Rouge / Jaune) ===

            print(f"Card classes: {CardDetector().classes}")

            card_transform = CardDetector().transform
            x = card_transform(image).unsqueeze(0).to(self.device)

            outputs, _, _, _, _ = self.card_model(x)
            probabilities = F.softmax(outputs, dim=1)
            confidence , predicted = torch.max(probabilities, 1)

            print(f"Fichier: {image_path}")
            print(f"Indice prédit: {predicted.item()}")
            print(f"Confiance: {confidence.item():.4f}")

            # Donc l'indice 0 = red , l'indice 1 = jaune
            if predicted.item() == 0:
                print("red")
                predictions.append(LABELS_INDEXES_BY_NAME["Red-Cards"])
            else:  
                print("yellow")
                predictions.append(LABELS_INDEXES_BY_NAME["Yellow-Cards"])

        return predictions

gm = GlobalModel()
gm.predict(["dataset/train/yellow_card/Yellow Cards__2__653.jpg"])