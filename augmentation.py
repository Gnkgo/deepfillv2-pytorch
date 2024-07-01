import numpy as np
import imgaug.augmenters as iaa
import cv2
import os
from glob import glob

# Define the augmentation sequences
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Geometric transformations applied to both image and mask
geo_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    sometimes(iaa.Crop(percent=(0, 0.1))),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)

# Color transformations applied only to the image
color_seq = iaa.Sequential([
    sometimes(iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2)
], random_order=True)

# Function to apply augmentation to two images simultaneously and save results
def augment_and_save(image1, image2, output_img_dir1, output_img_dir2, img_prefix, count=5):
    for i in range(count):
        # Apply the geometric augmentation sequence
        deterministic_geo_seq = geo_seq.to_deterministic()

        # Apply the geometric augmenter to both images and the mask
        augmented_image1 = deterministic_geo_seq(image=image1)
        augmented_image2 = deterministic_geo_seq(image=image2)
        
        # Apply the color transformations only to the image
        augmented_image1 = color_seq(image=augmented_image1)
        
        # Save augmented images in respective directories
        img_filename1 = os.path.join(output_img_dir1, f"{img_prefix}_{i}.png")
        img_filename2 = os.path.join(output_img_dir2, f"{img_prefix}_{i}.png")
        cv2.imwrite(img_filename1, augmented_image1)
        cv2.imwrite(img_filename2, augmented_image2)

# Function to process pairs of images from two folders
def process_folders(input_img_dir1, input_img_dir2, output_img_dir1, output_img_dir2, count=5):
    if not os.path.exists(output_img_dir1):
        os.makedirs(output_img_dir1)
    if not os.path.exists(output_img_dir2):
        os.makedirs(output_img_dir2)
    
    img_files1 = sorted(glob(os.path.join(input_img_dir1, "*.png")))
    img_files2 = sorted(glob(os.path.join(input_img_dir2, "*.png")))

    for img_file1 in img_files1:
        img_prefix = os.path.splitext(os.path.basename(img_file1))[0]
        img_file2 = os.path.join(input_img_dir2, f"{img_prefix}.png")

        if os.path.exists(img_file2):
            image1 = cv2.imread(img_file1)
            image2 = cv2.imread(img_file2)
            augment_and_save(image1, image2, output_img_dir1, output_img_dir2, img_prefix, count)

# Example usage
input_img_directory1 = 'images'
input_img_directory2 = 'masks'
output_img_directory1 = 'augmented_images'
output_img_directory2 = 'augmented_masks'

process_folders(input_img_directory1, input_img_directory2, output_img_directory1, output_img_directory2)
