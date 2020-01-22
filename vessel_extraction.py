import nibabel as nib
import os, glob
import numpy as np
import scipy.ndimage.interpolation
from skimage import measure, morphology
from skimage.filters import roberts, sobel
from skimage.feature import peak_local_max
import vtk_nifit_render
# import vtk_render_from_array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imsave

vesselmaskpath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/IPF21_20110321_vessel.nii'
lungmaskpath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/IPF21_20110321_ptk.nii'
niftipath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/IPF21_20110321_supine_3_3_B.nii'

# vesselmaskpath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/IPF21_20110321_vessel.nii'
# lungmaskpath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/IPF21_20171106.nii'
# niftipath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/IPF21_20171106_supine_3_2_B41f.nii'


# lungmaskpath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/AGING031ptk.nii'
# niftipath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/AGING031.nii'

# vess_im_nii = nib.load(vesselmaskpath)
# vess_im = np.asarray(vess_im_nii.get_data())

mask_im_nii = nib.load(lungmaskpath)
mask_im = np.asarray(mask_im_nii.get_data())


nifti_im_nii = nib.load(niftipath)
nifti_im = np.asarray(nifti_im_nii.get_data())
# nifti_im = np.fliplr(nifti_im) #for aging subjects alone

eroded_mask = scipy.ndimage.binary_erosion(mask_im, structure=np.ones((5, 5 , 5)))
mask_im = eroded_mask * mask_im

lung_only = np.where(mask_im > 0, nifti_im, 0)
masking_array = np.zeros(lung_only.shape)

def interpolate_inbetween(array_to_interpolate):
    ct = nifti_im_nii
    mask = mask_im_nii
    pixel_array = nifti_im
    pixel_array_mask = mask_im
    head = ct.get_header()
    xdim = head['dim'][1]
    ydim = head['dim'][2]
    zdim = abs(int(head['pixdim'][3]))
    factor = float(head['pixdim'][1])
    print xdim,ydim,zdim,factor

    print ct.shape
    print pixel_array.shape,pixel_array_mask.shape

    zoom_factor = [factor,factor,1]

    new_image = scipy.ndimage.interpolation.zoom(array_to_interpolate,zoom_factor,mode='nearest')
    # new_mask = scipy.ndimage.interpolation.zoom(pixel_array_mask,zoom_factor,mode='nearest')
    print new_image.shape
    return new_image

# new_image = interpolate_inbetween(lung_only)
# nifti_image_from_array = nib.Nifti1Image(new_image, nifti_im_nii.affine)
# save_path = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/tempnewimglungonly.nii'
# nib.save(nifti_image_from_array, save_path)

def segment_vessel_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want

    binary_image = np.where(image > -320, image , 0)

    binary_image[np.nonzero(binary_image)] = 1
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]
    # Fill the air around the person
    binary_image[background_label == labels] = 2
    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1
    vessel_extension = cc_thresholding(binary_image)
    vessel_extension[np.nonzero(vessel_extension)] = 1
    vessel_extension = vessel_extension * 255

    return vessel_extension

def cc_thresholding (mask):
    label_im, nb_labels = scipy.ndimage.label(mask)
    unique, counts, indices = np.unique(label_im, return_counts=True, return_index=True)
    # print unique, counts, indices
    threshold_labels = []
    for k in range(0, len(unique)):
        if indices[k] > 25:
            threshold_labels.append(unique[k])

    # print threshold_labels

    new_mask = np.where(np.isin(label_im, threshold_labels), label_im, 0)

    return new_mask

def centerline_extraction (mask):
    from skimage import morphology
    out_skeletonize = morphology.skeletonize_3d(mask)
    return out_skeletonize

binary_img = segment_vessel_mask(lung_only)

# imsave(("2dvessel.jpg"), binary_img[:,:,51])

binary_img = centerline_extraction(binary_img)


nifti_image_from_array = nib.Nifti1Image(binary_img, nifti_im_nii.affine)
save_path = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/temp.nii'
nib.save(nifti_image_from_array, save_path)
vtk_nifit_render.vtk_pipeline(save_path)
vtk_nifit_render.vtk_pipeline(vesselmaskpath)

print nifti_im_nii.affine

