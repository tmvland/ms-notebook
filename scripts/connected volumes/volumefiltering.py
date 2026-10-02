import cv2
import nibabel as nib
import numpy as np
import cc3d
from skimage.measure import regionprops

# 1. Load the NIfTI volume
nii_img = nib.load("subtracted_output.nii.gz")
volume = nii_img.get_fdata().astype(np.uint8)  # Ensure data is 8-bit binary (0 and 1 or 255)
affine = nii_img.affine
inv_affine = np.linalg.inv(affine)

voxel_sizes = nii_img.header.get_zooms()
print(f'Voxel sizes (mm): {voxel_sizes}')


labels_out = cc3d.connected_components(volume)

stats = regionprops(labels_out)

# Define your minimum size threshold (in mm)
min_size_threshold_mm = 7
min_ht_threshold_mm = 7
min_wd_threshold_mm = 7
min_z_threshold_mm = 7

min_size_threshold = min_size_threshold_mm/voxel_sizes[0]
min_ht_threshold = min_ht_threshold_mm/voxel_sizes[0]
min_wd_threshold = min_wd_threshold_mm/voxel_sizes[0]
min_z_threshold = min_z_threshold_mm/voxel_sizes[0]

# Define thresholds
min_voxel_size = 10
min_length_z, min_length_y, min_length_x = min_z_threshold, min_ht_threshold, min_wd_threshold

#FOR 3D ONLY---
filtered_volume = np.zeros_like(labels_out)

for region in stats:
    # Size (voxel count)
    voxel_count = region.area

    # Length (bounding box length along Z, Y, X axes)
    min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
    length_z = max_z - min_z
    length_y = max_y - min_y
    length_x = max_x - min_x

    # Apply filters
    size_ok = min_voxel_size <= voxel_count 
    length_ok = (
        length_z >= min_length_z
        and length_y >= min_length_y
        and length_x >= min_length_x
    )

    if size_ok and length_ok:
        # Keep the component
        filtered_volume[labels_out == region.label] = 1


# FOR 2D ONLY---
# filtered_volume = np.zeros_like(volume)

# for z in range(volume.shape[2]):
#     slice_2d = volume[:, :, z]

#     # Find all connected components
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         slice_2d, connectivity=8
#     )

#     # Create a blank canvas for the cleaned slice
#     cleaned_slice = np.zeros_like(slice_2d)

#     # Loop through all found components (skipping background index 0)
#     for i in range(1, num_labels):
#         size = stats[i, cv2.CC_STAT_AREA]
#         width = stats[i, cv2.CC_STAT_WIDTH]
#         height = stats[i, cv2.CC_STAT_HEIGHT]

#         # Keep only components larger than the threshold
#         if size >= min_size_threshold:
#             if width >= min_wd_threshold and height >= min_ht_threshold :
#                 cleaned_slice[labels == i] = 1  

#     filtered_volume[:, :, z] = cleaned_slice

# 3. Save the cleaned volume back to a NIfTI file
new_nii = nib.Nifti1Image(filtered_volume, nii_img.affine, nii_img.header)
nib.save(new_nii, "cleaned_mask.nii.gz")