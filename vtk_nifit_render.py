#Author: Joyce John
#This code reads in nifti data and does 3d volume rendering using header information
#This is useful for visualization

import nibabel as nib
import os, glob
import numpy as np
import vtk
from vtk.util import numpy_support

vesselmaskpath = '/hpc/jjoh182/Python Project Master/pulmonary_vessel/IPF21_20110321_vessel.nii'

vess_im_nii = nib.load(vesselmaskpath)
vess_im = np.asarray(vess_im_nii.get_data())
array_shape = vess_im.shape
print vess_im_nii.header
# VTK_data = numpy_support.numpy_to_vtk(num_array=vess_im.ravel(),deep=True,array_type=vtk.VTK_FLOAT)


# # PNG files path source (stack of images)
# files=glob.glob(r"/hpc/jjoh182/Python Project Master/pulmonary_vessel/Masks/*.tif")
#
# # Setting the file path
# filePath = vtk.vtkStringArray()
# # Sorting file to arrange in ascending order to get slices correctly
# files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
# filePath.SetNumberOfValues(len(files))
#
# for i in range(0,len(files),1):
#     filePath.SetValue(i,files[i])
#     # print(files[i])
def vtk_pipeline(vesselmaskpath):
    # 1. Source -Reader

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(vesselmaskpath)
    reader.Update()
    size = reader.GetOutput().GetDimensions()
    center = reader.GetOutput().GetCenter()
    spacing = reader.GetOutput().GetSpacing()
    reader.SetDataSpacing(spacing)





    # reader=vtk.vtkTIFFReader()
    # # reader=VTK_data
    # reader.SetFileNames(filePath)
    # reader.SetDataSpacing(0.63867188,  0.63867188,  3)
    reader.Update()

    print(reader)

    # 2. Filter --&gt; Setting the color mapper, Opacity for VolumeProperty
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(1, 1, 0.0, 0.0) # Red

    # To set different colored pores
    # colorFunc.AddRGBPoint(2, 0.0, 1, 0.0) # Green
    #colorFunc.AddRGBPoint(3, 0.0, 0, 1.0) # Black
    #colorFunc.AddRGBPoint(4, 0.0, 0.0, 1) # Blue

    opacity = vtk.vtkPiecewiseFunction()
    # opacity.AddPoint(1, 1, 0.0, 0.0)
    # opacity.AddPoint(2, 0.0, 0.0, 0.0)

    # The previous two classes stored properties and we want to apply
    # these properties to the volume we want to render,
    # we have to store them in a class that stores volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    # set the color for volumes
    volumeProperty.SetColor(colorFunc)
    # To add black as background of Volume
    volumeProperty.SetScalarOpacity(opacity)
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.SetIndependentComponents(2)

    #Ray cast function know how to render the data
    volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    # volumeMapper = vtk.vtkSmartVolumeMapper()
    # volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    # volumeMapper = vtk.vtkUnstructuredGridVolumeRayCastMapper()

    volumeMapper.SetInputConnection(reader.GetOutputPort())
    volumeMapper.SetBlendModeToMaximumIntensity()

    # Different modes are available in vtk for Blend mode functions
    #volumeMapper.SetBlendModeToAverageIntensity()
    #volumeMapper.SetBlendModeToMinimumIntensity()
    #volumeMapper.SetBlendModeToComposite()
    #volumeMapper.SetBlendModeToAdditive()

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ren = vtk.vtkRenderer()
    ren.AddVolume(volume)
    #No need to set by default it is black
    ren.SetBackground(0, 0, 0)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(900, 900)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)

    interactor.Initialize()
    renWin.Render()
    interactor.Start()
    return

# vtk_pipeline(vesselmaskpath)