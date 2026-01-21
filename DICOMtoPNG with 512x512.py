import os
import pydicom
import numpy as np
from PIL import Image
# from datasets import load_dataset

# # Login using e.g. huggingface-cli login to access this dataset
# ds = load_dataset("BTX24/tekno21-brain-stroke-dataset-binary")

def apply_window(image, center, width):
    """Apply window center/width to pixel data."""
    img_min = center - width / 2
    img_max = center + width / 2
    windowed = np.clip(image, img_min, img_max)
    windowed = ((windowed - img_min) / (img_max - img_min)) * 255.0
    return windowed.astype(np.uint8)

def dcm_to_png_with_resize(dcm_path, png_path, default_center=40, default_width=80, target_size=(512, 512)):
    """Convert DICOM to PNG, applying windowing and resizing."""
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(np.float32)
    
    # Try to get WC/WW
    center = ds.get('WindowCenter', default_center)
    width = ds.get('WindowWidth', default_width)

    if isinstance(center, pydicom.multival.MultiValue):
        center = center[0]
    if isinstance(width, pydicom.multival.MultiValue):
        width = width[0]

    # If WC/WW are suspicious, use default
    if width < 10 or width > 2000:
        center = default_center
        width = default_width

    pixel_array = apply_window(pixel_array, center, width)

    img = Image.fromarray(pixel_array)

    # Resize to target size
    img = img.resize(target_size, Image.LANCZOS)  # Good quality resize

    img.save(png_path)

def batch_convert_dicom_folder(dicom_folder, output_folder, default_center=40, default_width=80, target_size=(512, 512)):
    """Convert all DICOMs in a folder to resized PNGs."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(dicom_folder):
        if filename.lower().endswith(".dcm"):
            dcm_path = os.path.join(dicom_folder, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_folder, png_filename)
            
            try:
                dcm_to_png_with_resize(dcm_path, png_path, default_center, default_width, target_size)
                print(f"Converted: {filename} -> {png_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Example usage:
# batch_convert_dicom_folder('path_to_dicom_folder', 'path_to_output_folder')
# batch_convert_dicom_folder('dicom_images', 'png_images')
# os.makedirs("png_images", exist_ok=True)
batch_convert_dicom_folder(
    r'C:\Users\saide\OneDrive\Desktop\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images\hemorrhage\DICOM',
    r'C:\Users\saide\OneDrive\Desktop\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images\hemorrhage\PNG'
)

batch_convert_dicom_folder(
    r'C:\Users\saide\OneDrive\Desktop\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images\ischemic\DICOM',
    r'C:\Users\saide\OneDrive\Desktop\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images\ischemic\PNG'
)

# 3. Save all training images as PNG
for i, sample in enumerate(ds['train']):
    img = sample["image"]   # This is already a PIL image
    label = sample["label"] # 0 = normal, 1 = stroke
    
    img.save(f"png_images/{i}_label{label}.png")

print(" All images saved to png_images/")
# import os
# import pydicom
# import numpy as np
# from PIL import Image

# def apply_window(image, center, width):
#     img_min = center - width / 2
#     img_max = center + width / 2
#     windowed = np.clip(image, img_min, img_max)
#     windowed = ((windowed - img_min) / (img_max - img_min)) * 255.0
#     return windowed.astype(np.uint8)

# def dcm_to_png_with_resize(dcm_path, png_path, default_center=40, default_width=80, target_size=(512, 512)):
#     ds = pydicom.dcmread(dcm_path)
#     pixel_array = ds.pixel_array.astype(np.float32)
#     center = ds.get('WindowCenter', default_center)
#     width = ds.get('WindowWidth', default_width)
#     if isinstance(center, pydicom.multival.MultiValue):
#         center = center[0]
#     if isinstance(width, pydicom.multival.MultiValue):
#         width = width[0]
#     if width < 10 or width > 2000:
#         center = default_center
#         width = default_width
#     pixel_array = apply_window(pixel_array, center, width)
#     img = Image.fromarray(pixel_array)
#     img = img.resize(target_size, Image.LANCZOS)
#     img.save(png_path)

# def batch_convert_dicom_folders(input_root, output_root, default_center=40, default_width=80, target_size=(512, 512)):
#     for class_name in os.listdir(input_root):
#         class_input_folder = os.path.join(input_root, class_name)
#         class_output_folder = os.path.join(output_root, class_name)
#         if os.path.isdir(class_input_folder):
#             os.makedirs(class_output_folder, exist_ok=True)
#             for filename in os.listdir(class_input_folder):
#                 if filename.lower().endswith(".dcm"):
#                     dcm_path = os.path.join(class_input_folder, filename)
#                     png_filename = os.path.splitext(filename)[0] + ".png"
#                     png_path = os.path.join(class_output_folder, png_filename)
#                     try:
#                         dcm_to_png_with_resize(dcm_path, png_path, default_center, default_width, target_size)
#                         print(f"Converted: {dcm_path} -> {png_path}")
#                     except Exception as e:
#                         print(f"Failed to convert {dcm_path}: {e}")

# # Example usage:
# batch_convert_dicom_folders('dicom_images', 'png_images')

# batch_convert_dicom_folders(r'C:\Users\saide\OneDrive\Desktop\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images\ischemic\DICOM', 
#                            r'C:\Users\saide\OneDrive\Desktop\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images\hemmorrhage\DICOM')

