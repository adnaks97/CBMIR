from __future__ import print_function, division
import matplotlib
matplotlib.use("TKAgg")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
# import custom 
import nibabel as nib
import torchvision.models
# from torchvision.models.densenet import model_urls
from  scipy import misc

anatomy_list = ['BRAIN', 'Bladder', 'CBIS-DDSM', 'Colon', 'Eye', 'Kidney', 'Liver', 'Lung', 'MANDIBLE', 'PROSTRATE', 'Pancreas', 'Rectum', 'Thyroid', 'cervical', 'chest', 'esophagus', 'lymph', 'ovarian', 'phantom', 'renal', 'soft', 'stomach', 'uterus']
#anatomy_list = ['BRAIN','Eye','Liver','MANDIBLE','PROSTRATE','phantom'] # Uncomment this for 6 class and comment previous line
mode_6_class = '/media/bmi/MIL/skanda_25/skanda_25.pth'
model_23_class = '/media/bmi/MIL/stratified23_CORRECTED/stratified23_CORRECTED.pth'
trained_model = torch.load(model_23_class) # Load appropriate model
trained_model=trained_model.eval()
trained_model= trained_model.cuda()

trimmed_model = nn.Sequential(*list(trained_model.children())[:-1])
trimmed_model = trimmed_model.eval()
trimmed_model = trimmed_model.cuda()

origin = '/media/bmi/MIL/23_class_net/test/stomach/TCGA-VQ-A94R/TCGA-VQ-A94R_SCAN1_SLICE1.npy' # Paste the location of the Input SLICE here
path = '/' + ('/').join(origin.split('/')[1:-1])
print (path)

organ = path.split('/')[-2]

files =os.listdir(path)

i=0
for f in xrange(len(files)):
    if '.npy' in files[f]: 
        if i > 0:
            break
        print (str(i+1),'/',str(len(files)))
        print (path+'/'+files[f])
        # data = np.load(path+'/'+files[f])
        data = np.load(origin)
        anatomy = np.zeros((3,256,256))
        anatomy[0,:,:]= data
        anatomy[1:,:]= data
        anatomy[2,:,:]= data
        copy = anatomy
        # anatomy= np.swapaxes(anatomy,0,2)
        # anatomy= np.swapaxes(anatomy,0,1)


        trans           = transforms.ToTensor()
        T_Data          =trans(anatomy)

        T_Data  = T_Data.unsqueeze(0)

        v_t_data= Variable(T_Data)    
        c_output= trained_model(v_t_data.cuda())
        _,grade = torch.max(c_output,1)
        grade  =  grade.cpu().data.numpy()
        print (grade,anatomy_list[grade[0]])
        organ_class = anatomy_list[grade[0]]


        outputs = trimmed_model(v_t_data.cuda())
        outputs = outputs.cpu().data.numpy()
        outputs= outputs.reshape(1,512)


        i=i+1

loc1 = os.path.join('/media/bmi/MIL/codes/numpyFeatures_23_class',organ_class + '_training_representation.npy')
loc2 = os.path.join('/media/bmi/MIL/codes/numpyFeatures_23_class',organ_class + '_training_patient_list.npy')

without_class1 = '/media/bmi/MIL/codes/numpyFeatures_23_class/total_rep.npy'
without_class2 = '/media/bmi/MIL/codes/numpyFeatures_23_class/total_patients.npy'

closest_array= np.load(loc1) # IF USING WITHOUT CLASS PREDICTION replace 'loc1' with 'without_class1'
image_list   = np.load(loc2) # IF USING WITHOUT CLASS PREDICTION replace 'loc2' with 'without_class2'

error = closest_array- outputs

sq_error = np.power(error,2)

sq_error = np.sum(sq_error,axis=1)

msq_error = sq_error/512.0

idx = np.argsort(msq_error)

# print (idx,len(image_list),len(closest_array))

sorted_msq_error = msq_error[idx]

sorted_image_list = image_list[idx]


top_n_images= 11

closest_n_data_points = sorted_image_list[:top_n_images] 


def show_me_like_images (path,array_containing_image_names,top_n_images=11,copy=copy):
    counter =0
    for i in xrange(top_n_images+1):
        if i ==0:
            plt.subplot(2,6,i+1)
            copy= copy
            plt.imshow(copy[2],cmap='gray')
            plt.axis('off')
        else:
            plt.subplot(2,6,i+1)
            image= np.load(array_containing_image_names[counter])
            plt.imshow(image,cmap='gray')
            plt.axis('off')    
            counter= counter+1
    plt.subplots_adjust(wspace=0.04, hspace=0)
    save_loc = "/media/bmi/MIL/23_class_features_without_class" # ENTER FOLDER TO STORE IMAGE HERE
    plt.savefig(os.path.join(save_loc,organ + ".png"))
    plt.show()

# show_me_like_images('/media/brats/Varghese/swarna/train/Normal_liver',closest_n_data_points,top_n_images)
plt.imshow(copy[2],cmap='gray')
plt.axis('off')
plt.show()
show_me_like_images('/media/bmi/MIL/23_class_net/train/' + organ_class,closest_n_data_points,top_n_images,copy)


