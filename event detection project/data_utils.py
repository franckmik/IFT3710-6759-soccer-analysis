import os
import zipfile
import numpy as np
from PIL import Image # pillow

def extract_images(zip_path, label_name, label_index):
    images = []
    labels = []
    extract_path = zip_path.rstrip('.zip')  # Crée un dossier avec le même nom que le fichier zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    for file in os.listdir(extract_path+"/"+label_name):
        if file.endswith('.jpg') or file.endswith('.png'):
            file_path = os.path.join(extract_path, label_name, file)
            image = Image.open(file_path).convert('RGB')
            image = image.resize((64, 64))  # Redimensionner les images si nécessaire
            images.append(np.array(image))
            labels.append(label_index)
            image.close()  # Fermer l'image
            os.remove(file_path)  # Supprimer le fichier extrait pour éviter les duplications
    os.rmdir(extract_path+"/"+label_name) # Supprimer le dossier après traitement
    os.rmdir(extract_path)  # Supprimer le dossier après traitement
    return images, labels

def get_data(folder='train', events=[]):
    images, labels = [], []

    zip_files = [f"dataset/{folder}/" + event + ".zip" for event in events]

    for index, zip_file in enumerate(zip_files):
        #label = os.path.basename(zip_file).split('.')[0]  # Utiliser le nom du dossier ZIP comme label
        img, lbl = extract_images(zip_file, events[index], index)
        images.extend(img)
        labels.extend(lbl)

    # Convertir les données en tableaux numpy
    # images = np.array(images) / 255.0  # Normaliser les images
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

