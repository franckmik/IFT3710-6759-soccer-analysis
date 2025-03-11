import os
import zipfile
from PIL import Image  # pillow
import torch
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_images(folder_path, label_name, label_index):

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    for file in os.listdir(os.path.join(folder_path, label_name)):
        if file.endswith(('.jpg', '.png')):
            file_path = os.path.join(folder_path, label_name, file)
            try:
                with Image.open(file_path).convert('RGB') as image:
                    yield transform(image), label_index  # Générateur évitant le stockage en liste
            except Exception as e:
                print(file_path)
                print(f"Une erreur est survenue: {e}")

def get_data(folder, events=[]):
    images, labels = [], []

    folder_path = folder

    for index, event in enumerate(events):
        for image, label in extract_images(folder_path, event, index):
            images.append(image)
            labels.append(label)

    # Convertir directement en tenseurs (moins de copies en mémoire)
    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels
