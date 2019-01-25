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

#model_path = '/media/bmi/MIL/skanda_25/skanda_25.pth' # 6 class model path
model_path = '/media/bmi/MIL/stratified23_CORRECTED/stratified23_CORRECTED.pth' # LOAD APPROPRIATE MODE PATH
trained_model = torch.load(model_path)
trained_model=trained_model.eval()
trained_model=trained_model.cuda()

#anatomy_list = ['BRAIN', 'Eye', 'Liver', 'MANDIBLE', 'PROSTRATE', 'phantom'] # UNCOMMENT THIS LINE AND COMMENT BELOW LINE IF 6_CLASS_MODEL IS USED
anatomy_list = ['BRAIN', 'Bladder', 'CBIS-DDSM', 'Colon', 'Eye', 'Kidney', 'Liver', 'Lung', 'MANDIBLE', 'PROSTRATE', 'Pancreas', 'Rectum', 'Thyroid', 'cervical', 'chest', 'esophagus', 'lymph', 'ovarian', 'phantom', 'renal', 'soft', 'stomach', 'uterus']
d = dict()
for i in xrange(len(anatomy_list)):
    path = '/media/bmi/MIL/23_class_net/test'
    testing_path = os.path.join(path,anatomy_list[i])
    classes =i ### hit  the appropriate class
    numCases = 0
    counter =0
    # if anatomy_list[i] != 'chest':
    #     continue

    for case in os.listdir(testing_path):
        data = os.path.join(testing_path,case)

        files = os.listdir(data)
        numCases = numCases + len(files)
        for i in xrange(len(files)):

            # if i >10:
            #     break
            
            x = np.load(data+'/'+files[i])

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

            #print ('prediction---->',anatomy_list[pred[0]])

            d[anatomy_list[pred[0]]] = d.get(anatomy_list[pred[0]],0) + 1

            if pred == classes:
                counter= counter+1

    organ = testing_path.split('/')[-1]
    print
    print (organ)
    print ('predicted', str(counter),'/',str(numCases))
    print
    for val in d.keys():
        print (val,"----->",d[val])
    file_name = "Test_Results_MODEL-stratified23_CORRECTED.txt" # MODIFY THE FILE NAME APPROPRIATELY. FILE IS CREATED IN SAME LOCATION AS 'swarna_testing_MAIN.py'
    handle = open(file_name,"a")
    content = 'predicted : ' + str(counter) + '/' + str(numCases) + '\nAccuracy : ' + str(counter/float(numCases))
    handle.write(organ + "\n") 
    handle.write(content)
    handle.write("\n-------------------------------------\n")







