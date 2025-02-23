import os
import shutil
import pandas as pd

# Define paths
train_csv_path = "/Users/rajeevranjanpratapsingh/PycharmProjects/dataminig /aptos2019-blindness-detection/train.csv"  # Path to train.csv
train_images_folder = "/Users/rajeevranjanpratapsingh/PycharmProjects/dataminig /aptos2019-blindness-detection/train_images"  # Path to training images folder
output_base_folder = "classified_images"  # Path where classified folders will be created

# Load the CSV file
df = pd.read_csv(train_csv_path)

# Create directories for each class if they don't exist
class_folders = {label: os.path.join(output_base_folder, f"{label}_{desc}")
                 for label, desc in zip(range(5), ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"])}

for folder in class_folders.values():
    os.makedirs(folder, exist_ok=True)

# Move images into respective folders with a check
moved_count = 0
total_images = len(df)

for _, row in df.iterrows():
    image_filename = f"{row['id_code']}.png"  # Modify if file extension differs
    image_path = os.path.join(train_images_folder, image_filename)
    target_path = os.path.join(class_folders[row['diagnosis']], image_filename)

    if os.path.exists(image_path):  # Ensure the image file exists
        shutil.move(image_path, target_path)
        if os.path.exists(target_path):  # Double check the file moved correctly
            moved_count += 1
        else:
            print(f"Warning: {image_filename} was not moved correctly!")

print(f"Images successfully classified and moved: {moved_count}/{total_images}")
