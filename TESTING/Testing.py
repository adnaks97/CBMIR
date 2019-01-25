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
import custom
import nibabel as nib

import torchvision.models
# from torchvision.models.densenet import model_urls


trained_model = torch.load('/media/bmi/MIL/stratified23/stratified23.pth')
trained_model=trained_model.eval()
trained_model=trained_model.cuda()

testing_path = '/media/bmi/MIL/19_class_net/test/jnk'
classes =1 ### hit  the appropriate class

files = os.listdir(testing_path)
counter =0
for i in xrange(len(files)):

    # if i >10:
    #     break

    x = np.load(testing_path+'/'+files[i])

    anatomy = np.zeros((3,256,256))

    anatomy[0,:,:]=x
    anatomy[1,:,:]=x
    anatomy[2,:,:]=x

    # anatomy=np.swapaxes(anatomy,0,2)
    # anatomy= np.swapaxes(anatomy,0,1)


    trans           = transforms.ToTensor()
    T_Data          =trans(anatomy)

    T_Data  = T_Data.unsqueeze(0)

    v_t_data= Variable(T_Data)
    outputs = trained_model(v_t_data.cuda())

    _,pred  = torch.max(outputs,1)
        # pred    = pred.cpu().data.numpy()
    pred    = pred.cpu().data.numpy()

    # print ('prediction---->',pred)

    if pred == classes:
        counter= counter+1

print ('predicted', str(counter),'/',str(len(files)))
