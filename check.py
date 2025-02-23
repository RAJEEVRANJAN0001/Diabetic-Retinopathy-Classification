import os
import pandas as pd

# Define paths
train_csv_path = "/Users/rajeevranjanpratapsingh/PycharmProjects/dataminig /aptos2019-blindness-detection/train.csv"  # Path to train.csv
classified_images_folder = "/Users/rajeevranjanpratapsingh/PycharmProjects/dataminig /classified_images"  # Path to classified images folders
report_csv_path = "classification_report.csv"  # Output report file

# Load the CSV file
df = pd.read_csv(train_csv_path)

# Create a dictionary mapping image filename to its correct label
image_label_map = {f"{row['id_code']}.png": row['diagnosis'] for _, row in df.iterrows()}

# List all classified folders
class_folders = {label: f"{label}_{desc}" for label, desc in
                 zip(range(5), ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"])}

# Prepare report data
report_data = []

# Iterate through all images in classified folders
for label, folder_name in class_folders.items():
    folder_path = os.path.join(classified_images_folder, folder_name)

    if os.path.exists(folder_path):  # Ensure the folder exists
        for image_filename in os.listdir(folder_path):
            expected_label = image_label_map.get(image_filename, "Not Found")
            actual_label = label
            status = "Correct" if expected_label == actual_label else "Incorrect"

            report_data.append([image_filename, expected_label, actual_label, status])

# Convert report data to DataFrame
report_df = pd.DataFrame(report_data, columns=["image_filename", "expected_label", "actual_label", "status"])

# Save report as CSV
report_df.to_csv(report_csv_path, index=False)

print(f"Classification report saved as {report_csv_path}")
