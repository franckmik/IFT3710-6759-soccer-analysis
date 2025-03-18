import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_model.vae import VAE, vae_loss
from cards.card_classifier import CardClassifier
from my_module import SoccerEventClassifier


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


        self.vae_model = VAE().to(device)
        self.vae_model.load_state_dict(torch.load('vae_model/vae_model_upsample_4.pth', weights_only=True))
        self.vae_model.eval()

        self.card_model = CardClassifier().to(device)
        self.card_model.load_state_dict(torch.load('cards/card_model_color_enhanced.pth', weights_only=True))
        self.card_model.eval()

        self.image_classifier_model = SoccerEventClassifier().to(device)
        self.image_classifier_model.load_state_dict(torch.load('model_event_classifier.pth', weights_only=True))
        self.image_classifier_model.eval()


    def predict(self, x):
        """
        Effectue une prédiction en combinant les résultats des 3 modèles
        selon une logique conditionnelle personnalisée.
        """
        x = x.to(self.device)

        # Model VAE
        # si recons error > threshold, return no_highlight
        # sinon
        # Model image classification
        # si 'Left', 'Right' ou 'Center', return no highlight
        # sinon si 'Cards' -> model card
        # else return result
        # Model cards
        # return rouge ou jaune

        # No highlight detection step

        recon_x, mu, logvar = self.vae_model(x)

        loss = vae_loss(recon_x, x, mu, logvar)

        if loss >= (VAE_PAPER_THRESHOLD/2):
            return LABELS_INDEXES_BY_NAME["No-highlight"]


        # Image classification step

        image_classifier_outputs = self.image_classifier_model(x)
        _, predicted_index = torch.max(image_classifier_outputs.data, 1)

        if(predicted_index in [self.image_classifier_model.class_names_dict["Left"],
                               self.image_classifier_model.class_names_dict["Right"],
                               self.image_classifier_model.class_names_dict["Center"]]):
            return LABELS_INDEXES_BY_NAME["No-highlight"]

        if(predicted_index != self.image_classifier_model.class_names_dict["Cards"]):
            return LABELS_INDEXES_BY_NAME[self.image_classifier_model.class_names[predicted_index]]

        # Cards classification step

        outputs, _, _, _, _ = self.card_model(x)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

        if predicted == 0:
            return LABELS_INDEXES_BY_NAME["Yellow-Cards"]
        else:
            return LABELS_INDEXES_BY_NAME["Red-Cards"]