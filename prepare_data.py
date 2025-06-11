import os
import random
import shutil

print("--- Starting Data Preparation ---")

original_data_dir = 'ai_vs_real_images' 
if not os.path.exists(original_data_dir):
    print(f"ERROR: The directory '{original_data_dir}' was not found.")
    print("Please make sure you have unzipped the Kaggle data correctly.")
    exit()

base_dir = '.'
split_ratio = 0.8 # 80% train, 20% test

# Create main train and test directories
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in ['real', 'fake']:
    print(f"Processing class: {class_name}")
    
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)
    
    original_class_path = os.path.join(original_data_dir, class_name)
    filenames = os.listdir(original_class_path)
    random.shuffle(filenames)
    
    split_point = int(len(filenames) * split_ratio)
    train_files = filenames[:split_point]
    test_files = filenames[split_point:]
    
    print(f"  Copying {len(train_files)} files to train/{class_name}...")
    for fname in train_files:
        shutil.copyfile(os.path.join(original_class_path, fname), os.path.join(train_class_dir, fname))
        
    print(f"  Copying {len(test_files)} files to test/{class_name}...")
    for fname in test_files:
        shutil.copyfile(os.path.join(original_class_path, fname), os.path.join(test_class_dir, fname))

print("\n✅ Data preparation complete! You can now train your model.")