# from datasets import load_dataset

# # Login using e.g. huggingface-cli login to access this dataset
# ds = load_dataset('BTX24/tekno21-brain-stroke-dataset-binary')
# print(ds)
# example = ds["train"][0]
# print(example["label"])   # shows label
# example["image"].show()   # displays CT image if PIL is installed
# print(ds['train'][0])  # inspect one sample

# from PIL import Image


# # Open an image
# img = Image.open("brain_scan.png")

# # Show it
# img.show()

# # Resize it
# resized = img.resize((224, 224))

# # Convert to array (for ML)
# import numpy as np
# arr = np.array(resized)
# print(arr.shape)
import os
import shutil

img_folder = r"c:\Users\saide\OneDrive\Desktop\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images"

label_map = {
    "0": "non-stroke",
    "1": "stroke"
}

# Create subfolders
for class_name in label_map.values():
    os.makedirs(os.path.join(img_folder, class_name), exist_ok=True)

# Move files
for fname in os.listdir(img_folder):
    if fname.endswith(".png") and "label" in fname:
        label = fname.split("label")[1].split(".")[0]
        class_name = label_map.get(label)
        if class_name:
            src = os.path.join(img_folder, fname)
            dst = os.path.join(img_folder, class_name, fname)
            shutil.move(src, dst)

print("Images organized into 'non-stroke' and 'stroke' folders.")
