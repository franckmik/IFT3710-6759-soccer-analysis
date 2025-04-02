import os
import shutil
import random
import argparse

def move_random_images(source_dir, dest_dir, num_images=500, file_ext='.jpg'):
    """
    Déplace aléatoirement un nombre spécifié d'images d'un répertoire source vers un répertoire destination.
    
    Args:
        source_dir (str): Chemin du répertoire source
        dest_dir (str): Chemin du répertoire destination
        num_images (int): Nombre d'images à déplacer (par défaut 500)
        file_ext (str): Extension des fichiers à déplacer (par défaut '.jpg')
    
    Returns:
        int: Nombre d'images effectivement déplacées
    """
    # Vérifier que les répertoires existent
    if not os.path.isdir(source_dir):
        raise ValueError(f"Le répertoire source {source_dir} n'existe pas.")
    
    # Créer le répertoire de destination s'il n'existe pas
    os.makedirs(dest_dir, exist_ok=True)
    
    # Obtenir la liste de toutes les images jpg dans le répertoire source
    all_images = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(file_ext) and os.path.isfile(os.path.join(source_dir, f))]
    
    # Vérifier s'il y a suffisamment d'images dans le répertoire source
    available_images = len(all_images)
    if available_images == 0:
        print(f"Aucune image {file_ext} trouvée dans {source_dir}")
        return 0
    
    # Nombre d'images à déplacer (limité par le nombre disponible)
    num_to_move = min(num_images, available_images)
    
    # Sélectionner aléatoirement les images à déplacer
    selected_images = random.sample(all_images, num_to_move)
    
    # Compteur d'images déplacées
    moved_count = 0
    
    # Déplacer les images
    for img in selected_images:
        source_path = os.path.join(source_dir, img)
        dest_path = os.path.join(dest_dir, img)
        
        try:
            shutil.move(source_path, dest_path)
            moved_count += 1
            print(f"Déplacé: {img}")
        except Exception as e:
            print(f"Erreur lors du déplacement de {img}: {e}")
    
    print(f"\n{moved_count} images sur {num_images} demandées ont été déplacées avec succès.")
    return moved_count

if __name__ == "__main__":
    chemin_absolu = "C:\\Users\\herve\\OneDrive - Universite de Montreal\\Github\\IFT3710-6759-soccer-analysis\\event detection project\\dataset\\"
    
    # Appeler la fonction avec les arguments fournis
    move_random_images(chemin_absolu+"train\\red_card",chemin_absolu+"validation\\red_card", 500, ".jpg")
    move_random_images(chemin_absolu+"train\\yellow_card",chemin_absolu+"validation\\yellow_card", 500, ".jpg")
