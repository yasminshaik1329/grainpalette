import os
import shutil
import random

# Paths
original_data_dir = 'Rice_Image_Dataset'  # The folder with class subfolders
output_base_dir = 'rice_dataset'
train_ratio = 0.8  # 80% for training, 20% for testing

# Create train and test folders
for category in os.listdir(original_data_dir):
    category_path = os.path.join(original_data_dir, category)
    if os.path.isdir(category_path):
        images = os.listdir(category_path)
        random.shuffle(images)

        split_point = int(train_ratio * len(images))
        train_images = images[:split_point]
        test_images = images[split_point:]

        # Make new folders
        os.makedirs(os.path.join(output_base_dir, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, 'test', category), exist_ok=True)

        # Copy files
        for img in train_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(output_base_dir, 'train', category, img)
            shutil.copyfile(src, dst)

        for img in test_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(output_base_dir, 'test', category, img)
            shutil.copyfile(src, dst)

print("✅ Dataset successfully split into 'train' and 'test' folders.")
