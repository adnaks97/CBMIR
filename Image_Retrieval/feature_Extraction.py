####
'''
@uthor Skanda
written to create numpy array  and find closest points.
'''
####
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


trained_model = torch.load('/media/bmi/MIL/stratified23_CORRECTED/stratified23_CORRECTED.pth')
trained_model= trained_model.eval()
trained_model=trained_model.cuda()

trimmed_model = nn.Sequential(*list(trained_model.children())[:-1])

home = '/media/bmi/MIL/23_class_net/train'
for organ in os.listdir(home):
    save1 = "./numpyFeatures_23_class/" + organ + '_training_representation.npy'
    save2 = "./numpyFeatures_23_class/" + organ + '_training_patient_list.npy'
    if os.path.exists(save1):
        continue
    organ_path = os.path.join(home,organ)
    image_list = []
    file_count = 0
    i = 0
    for patient in os.listdir(organ_path):
        file_count = file_count + len(os.listdir(os.path.join(organ_path,patient)))
    print (organ)
    print("file_count = ",str(file_count))
    hidden_representation= np.zeros((file_count,512))
    for patient in os.listdir(organ_path):
        path = os.path.join(organ_path,patient)
        files =os.listdir(path)
        # counter =0
        # classes=1
        # i=0
        for f in xrange(len(files)):
            if '.npy' in files[f]:

                print (str(i+1),'/',str(file_count))
                # if i > 2:
                #     break
                data = np.load(path+'/'+files[f])
                image_list.append(path+'/'+files[f])
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
                outputs = trimmed_model(v_t_data.cuda())
                outputs = outputs.cpu().data.numpy()
                outputs= outputs.reshape(1,512)

                print (outputs.shape)
                print (patient,'----->',files[f])
                hidden_representation[i,:]= outputs
                i=i+1
    print ('saving the hidden_representation of',organ)
    np.save(save1,hidden_representation)
    np.save(save2,image_list)



#     _,pred  = torch.max(outputs,1)


#     pred    = pred.cpu().data.numpy()

#     if pred == classes:
#         counter= counter+1

# print ('predicted', str(counter),'/',str(len(files)))
