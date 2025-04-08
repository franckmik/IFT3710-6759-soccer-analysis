# augment_dataset.py
import os
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
from tqdm import tqdm


def augment_dataset(input_dir, output_dir, class_names):
    """
    Creates 4 versions of each image with different augmentations and
    preserves the original class structure, without color distortion.

    Args:
        input_dir (str): Path to input directory with original images
        output_dir (str): Path to output directory for augmented images
        class_names (list): List of class folder names
    """
    # Create the transformations without normalization to preserve colors
    scale_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    rotate_transform = transforms.Compose([
        transforms.Resize((248, 248)),  # Slightly larger to avoid black edges
        transforms.RandomRotation(15),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])

    shift_transform = transforms.Compose([
        transforms.Resize((248, 248)),  # Larger to allow for translation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])

    flip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),  # 100% chance to flip
        transforms.ToTensor()
    ])

    # Process each class
    for class_name in class_names:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)

        # Create output directory for this class
        os.makedirs(output_class_dir, exist_ok=True)

        # Skip if input directory doesn't exist
        if not os.path.exists(input_class_dir):
            print(f"Warning: Class directory not found: {input_class_dir}")
            continue

        print(f"Processing class: {class_name}")

        # Process each image in this class
        image_files = [f for f in os.listdir(input_class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in tqdm(image_files, desc=f"Augmenting {class_name}"):
            # Extract name and extension
            name, ext = os.path.splitext(img_file)

            # Load the image
            img_path = os.path.join(input_class_dir, img_file)
            try:
                original_img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

            try:
                # Create and save scaled version (without normalization)
                scale_img = scale_transform(original_img)
                save_image(scale_img, os.path.join(output_class_dir, f"scale_{name}{ext}"))

                # Create and save rotated version (without normalization)
                rotate_img = rotate_transform(original_img)
                save_image(rotate_img, os.path.join(output_class_dir, f"rotate_{name}{ext}"))

                # Create and save shifted version (without normalization)
                shift_img = shift_transform(original_img)
                save_image(shift_img, os.path.join(output_class_dir, f"shift_{name}{ext}"))

                # Create and save flipped version (without normalization)
                flip_img = flip_transform(original_img)
                save_image(flip_img, os.path.join(output_class_dir, f"flip_{name}{ext}"))

                # Optional: Save original image as well (without normalization)
                orig_tensor = scale_transform(original_img)
                save_image(orig_tensor, os.path.join(output_class_dir, f"orig_{img_file}"))

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

    print(f"Augmentation complete! Augmented dataset saved to: {output_dir}")


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Create augmented dataset for card classification')
    parser.add_argument('--input_dir', type=str,
                        default=None,
                        help='Path to input directory containing original images')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Path to output directory for augmented images')
    parser.add_argument('--classes', nargs='+', type=str,
                        default=["red_card", "yellow_card"],
                        help='List of class folder names (default: red_card yellow_card)')

    args = parser.parse_args()

    # Default paths if not provided
    if args.input_dir is None or args.output_dir is None:
        # Default path from your project
        chemin_absolu = "C:\\Users\\herve\\OneDrive - Universite de Montreal\\Github\\IFT3710-6759-soccer-analysis\\event detection project\\dataset\\"

        if args.input_dir is None:
            args.input_dir = chemin_absolu + "train"

        if args.output_dir is None:
            args.output_dir = chemin_absolu + "augmented_train"

    # Validate input directory
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Classes: {args.classes}")

    # Run the augmentation
    augment_dataset(args.input_dir, args.output_dir, args.classes)

    # Count original and augmented images
    orig_count = sum(len([f for f in os.listdir(os.path.join(args.input_dir, cls))
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                     for cls in args.classes if os.path.exists(os.path.join(args.input_dir, cls)))

    aug_count = sum(len([f for f in os.listdir(os.path.join(args.output_dir, cls))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    for cls in args.classes if os.path.exists(os.path.join(args.output_dir, cls)))

    print(f"Original dataset: {orig_count} images")
    print(f"Augmented dataset: {aug_count} images")
    print(f"Augmentation factor: {aug_count / orig_count if orig_count > 0 else 'N/A'}")


if __name__ == "__main__":
    main()