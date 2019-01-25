# Imports
from __future__ import division
import numpy as np
import os, sys, shutil, re
import h5py
import random, time
import scipy.ndimage as snd
import skimage.morphology as morph
import random
from datetime import datetime
import glob
import cPickle as pickle
from collections import namedtuple
import skimage.transform
import skimage.draw
import skimage.morphology as morph
from skimage.feature import canny
import matplotlib.pyplot as plt
from cv2 import bilateralFilter
from skimage.transform import resize
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas  as pd
import nibabel as nib

def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage: 
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
    plt.show()

def resize_sitk_2D(image_array, outputSize=None, interpolator=sitk.sitkLinear):
    """
    Resample 2D images Image:
    For Labels use nearest neighbour
    For image use 
    sitkNearestNeighbor = 1,
    sitkLinear = 2,
    sitkBSpline = 3,
    sitkGaussian = 4,
    sitkLabelGaussian = 5, 
    """
    image = sitk.GetImageFromArray(image_array) 
    inputSize = image.GetSize()
    inputSpacing = image.GetSpacing()
    outputSpacing = [1.0, 1.0]
    if outputSize:
        outputSpacing[0] = inputSpacing[0] * (inputSize[0] /outputSize[0]);
        outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1]);
    else:
        # If No outputSize is specified then resample to 1mm spacing
        outputSize = [0.0, 0.0]
        outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
        outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(outputSize)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    image = resampler.Execute(image)
    resampled_arr = sitk.GetArrayFromImage(image)
    return resampled_arr

def resize_sitk_3D(image_array, outputSize=None, interpolator=sitk.sitkLinear):
    """
    Resample 3D images Image:
    For Labels use nearest neighbour
    For image use 
    sitkNearestNeighbor = 1,
    sitkLinear = 2,
    sitkBSpline = 3,
    sitkGaussian = 4,
    sitkLabelGaussian = 5, 
    """
    image = sitk.GetImageFromArray(image_array) 
    inputSize = image.GetSize()
    inputSpacing = image.GetSpacing()
    outputSpacing = [1.0, 1.0, 1.0]
    if outputSize:
        outputSpacing[0] = inputSpacing[0] * (inputSize[0] /outputSize[0]);
        outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1]);
        outputSpacing[2] = inputSpacing[2] * (inputSize[2] / outputSize[2]);
    else:
        # If No outputSize is specified then resample to 1mm spacing
        outputSize = [0.0, 0.0, 0.0]
        outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
        outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
        outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(outputSize)
    resampler.SetOutputSpacing(outputSpacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    image = resampler.Execute(image)
    resampled_arr = sitk.GetArrayFromImage(image)
    return resampled_arr

def HU_window(in_arr, low=0, high=400):
    return np.clip(in_arr, low, high)

def histogram_equalization(arr):
    nbr_bins = 256
    imhist, bins = np.histogram(arr.flatten(), nbr_bins, normed = True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    original_shape = arr.shape
    arr = np.interp(arr.flatten(), bins[:-1], cdf)
    out_arr = arr.reshape(original_shape)
    return out_arr

def normalize(x, _max=255, _min=0):
    x = np.float32(x)
    min_val = np.min(x)
    max_val = np.max(x)
    ret_val = ((x - min_val) / (max_val - min_val))*(_max - _min) + _min
    return np.uint8(np.round(ret_val))


if __name__ == '__main__':
    outputSize = (256, 256)
    # Path to your data
    # path = './lesion_liver'
    # anatomy_list = ['lesion_brain_MR.csv', 'lesion_liver_MR.csv', 'normal_brain_MR.csv', 'normal_liver_CT.csv', 'normal_brain_CT.csv']
    # csv_path ='/home/bmi/Desktop/Deep-retrieval/csv files'
    root = '/media/bmi/MIL/numpyFiles'
    # save_path ='/home/bmi/Desktop/Deep-retrieval/processed'

    anatomy_list = os.listdir(root)

    # csv_path ='/home/bmi/Desktop/Deep-retrieval/New_test _images'

    for file in anatomy_list:
        if '.' in file:
            anatomy_list.remove(file)




    #     csv_read = pd.read_csv(os.path.join(csv_path, anatomy))
    #     File_names = csv_read['File name']
    #     Starts     = csv_read['Start']
    #     Stops     = csv_read['stop']
    #     number_of_volumes = len(Stops)
    #     inc = 0
    #     anatomy_name = anatomy.split('.')[0]
    #     modality = anatomy.split('.')[0].split('_')[2]            save_location = os.path.join(d
    #     # Path to your destination 
    #     for i in xrange(number_of_volumes):

    #         #
    #         # if i > 0:
    #         #     break
    #         print ('patient id', File_names[i])

    #         print ('Start:',Starts[i])

    #         print ('Stop:',Stops[i])
    #         #
    #         #
    #         patient_volume = nib.load(File_names[i])
    #         #
    #         patient_volume_data= patient_volume.get_data()
    #         for sl in range(Starts[i]-1,Stops[i]):
    #             # print sl
    #         slices = np.rot90(patient_volume_data[:,:,int(sl)])
    #             save_location = os.path.join(dest_path, modality+anatomy_name+str(inc)+'.npy')
                # Preprocess
    dest_root = '/media/bmi/MIL/numpyFiles_resized'
    for anatomy in anatomy_list:
        if 'junk' in anatomy:
            continue
        # if anatomy in ['CBIS-DDSM']: #MODIFY THIS BLOCK AS REQUIRED
        #     continue
        anatomy_path = os.path.join(root,anatomy)
        patients = os.listdir(anatomy_path)
        dest_anatomy_path = os.path.join(dest_root, anatomy + '_resized')
        if not os.path.exists(dest_anatomy_path):
            os.makedirs(dest_anatomy_path)
        else:
            continue
        for patient in patients:
            patient_path = os.path.join(anatomy_path,patient)
            numpyFiles = os.listdir(patient_path)            
            for file in numpyFiles:
                print file
                file_path = os.path.join(patient_path,file)
                print file_path
                slices = np.load(file_path)
                slices_resized = slices
                if slices.shape != outputSize:
                    slices_resized = resize_sitk_2D(slices, outputSize)
                print slices_resized.shape
                print slices_resized[0,0]
                print
                if slices_resized[0,0] < 0:
                    slices_resized = HU_window(slices_resized)
                    print "HU"
                #print 'dest_path : ', (dest_path)
                dest_path = os.path.join(dest_anatomy_path,patient)
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                save_location = os.path.join(dest_path, file )  
                print save_location        
                # data_resized_HU_hist_eq = histogram_equalization(data_resized_HU)
                slices = normalize(slices) 
                np.save(save_location, slices_resized)
                a = np.load(save_location)
                print a.shape
                print np.min(a),np.max(a)

                #inc= inc+1
                data = slices
                data_resized = slices_resized
                #imshow(data, data_resized, title = [data.shape, data_resized.shape, 'Difference Image', 'Normalized']) 



        