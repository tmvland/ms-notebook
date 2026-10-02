#generalized cor/sag/ax code
import nibabel as nib
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import re
from nibabel.processing import resample_from_to
import skimage as ski
import os
from pathlib import Path
import ants
import cv2
from scipy.ndimage import affine_transform
os.environ['ITK_NIFTI_SFORM_PERMISSIVE'] = '1'


# 1. Prompt the user and save the string into a variable
file_path = input("Enter path to data: ")

# Optional clean-up: Remove trailing spaces or accidental surrounding quotes
file_path = file_path.strip("'\"")

# 2. Verify if the file path actually exists
if os.path.exists(file_path):
    print(f"Success! File found at: {file_path}")
    # Proceed with your logic (e.g., open and read the file)
else:
    print(f"Error: The path '{file_path}' does not exist.")

file_path2 = input("Enter path to data: ")

# Optional clean-up: Remove trailing spaces or accidental surrounding quotes
file_path2 = file_path2.strip("'\"")

# 2. Verify if the file path actually exists
if os.path.exists(file_path2):
    print(f"Success! File found at: {file_path2}")
    # Proceed with your logic (e.g., open and read the file)
else:
    print(f"Error: The path '{file_path2}' does not exist.")


user_entry = file_path
ref_img = nib.load(file_path)
ref_shape = ref_img.shape
user_entry1 = file_path2
affine_matrix = ref_img.affine


parts = file_path.split('.')
# tag = parts[0]
tag = os.path.basename(user_entry) 
tag = (Path(tag).stem)  

#load txt instructions

with open(user_entry1, "r", encoding="utf-8") as file:
    content = file.read()
    # print(content)

pattern = r"(Begin Irregular ROI.*?End Irregular ROI)"
# Find all matching ROI blocks
rois = re.findall(pattern, content, re.DOTALL)
pattern = r"(Begin Spline ROI.*?End Spline ROI)"
# Find all matching ROI blocks
rois = rois + re.findall(pattern, content, re.DOTALL)

def maskvol(img,rois,view):
    
    empty_data = img.copy()
    empty_data = np.zeros(empty_data.shape)

    m = 0
    text = rois[m]
    match = re.search(r"Slice=(\d+)", text)
    slicenum = int(match.group(1))
    

    while m < len(rois):
        # print(slicenum)
        text = rois[m]

        match = re.search(r"Slice=(\d+)", text)
        slicenum = int(match.group(1))
        if match:

            x_roi = re.findall(r'X=([-+]?\d*\.?\d+)', text)
            y_roi = re.findall(r'Y=([-+]?\d*\.?\d+)', text)

            x_vox = (empty_data.shape[0]/2)
            y_vox = (empty_data.shape[1]/2)
            # x_vox = 0
            # y_vox = 0

            if view == 'A':
                roi_voxels = np.array([
                                    [(x_vox-float(x)) for x in x_roi],
                                    [(y_vox-float(y)) for y in y_roi],
                                    [slicenum for y in y_roi]
                                    ])

            if view == 'C':
                roi_voxels = np.array([
                                        [(x_vox-float(y)) for y in y_roi],
                                        [(y_vox-float(x)) for x in x_roi],
                                        [slicenum for y in y_roi]
                                                    ])

            roi_voxels = np.round(roi_voxels).astype(int)
            x_array = roi_voxels[0]
            y_array = roi_voxels[1]
                
            mask = np.ones(img.shape[:2], dtype=bool)

            x1 = max(x_array)
            x2 = min(x_array)

            y1 = max(y_array)
            y2 = min(y_array)

            # rr, cc = ski.draw.rectangle(start=(x2, y2), end=(x1, y1))
            rr, cc = ski.draw.polygon(x_array,y_array,mask.shape)

            mask[rr,cc] = False
            # mask[roi_voxels[0], roi_voxels[1]] = False
            img_masked = img[:,:,slicenum].copy()
            img_masked[mask] = 0  
            empty_data[:,:,slicenum] = img_masked + empty_data[:,:,slicenum]
        m+=1
    return(empty_data)

original_affine = ref_img.affine

#Example usage:

new_mask = maskvol(ref_img.get_fdata(),rois,'A')

new_img = nib.Nifti1Image(new_mask,affine = original_affine)
nib.save(new_img, 'new_mask.nii.gz')



