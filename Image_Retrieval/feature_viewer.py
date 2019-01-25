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

model = torch.load('/media/bmi/MIL/stratified23_CORRECTED/stratified23_CORRECTED.pth')
model.cuda()

my_model = nn.Sequential(*list(model.children())[:1])



data = np.load('/media/bmi/MIL/23_class_net/test/BRAIN/Brats17_TCIA_640_1/Brats17_TCIA_640_1_33.npy')
## mr_normal_brain = /media/brats/Varghese/swarna/Test_set/Normal_brain/MRnormal_brain_MR278.npy
## ct_normal_brain = /media/brats/Varghese/swarna/Test_set/Normal_brain/CTnormal_brain_CT366.npy
## lesion liver= /media/brats/Varghese/swarna/Test_set/Lesion_liver/MRlesion_liver_MR34.npy
## lesion_brain =/media/brats/Varghese/swarna/Test_set/Lesion_brain/MRlesion_brain_MR210.npy
## normal_liver =
anatomy = np.zeros((3,256,256))
anatomy[0,:,:]=data
anatomy[1,:,:]=data
anatomy[2,:,:]=data
copy= anatomy
# anatomy    =np.swapaxes(anatomy,0,2)
# anatomy    = np.swapaxes(anatomy,0,1)

trans    = transforms.ToTensor()
T_Data          =trans(anatomy)

T_Data  = T_Data.unsqueeze(0)
T_Data  = Variable(T_Data)
T_Data = T_Data.cuda()
outs= my_model(T_Data)

outs= outs.data.cpu().numpy()


def show_me_feature_maps(feature_map,num_of_rows=8, num_of_columns=8, see_all_feature_map=False):
	one_feature_map= feature_map[0]

	num_of_feature_maps =32   ###3 to visualize limited number of feature map

	if see_all_feature_map== True:
		num_of_feature_maps= one_feature_map.shape[0]
		num_of_rows= num_of_feature_maps/num_of_columns
		print (num_of_rows)



	for i in range(1,num_of_feature_maps+1):
		plt.subplot(num_of_rows,num_of_columns,i)
		plt.imshow(misc.imresize(one_feature_map[i-1],[256,256]),cmap='gray')
	plt.show()

def show_me_individual_feature_maps(feature_map,feature_map_id=3):
	one_feature_map= feature_map[0]

	num_of_feature_maps =32   ###3 to visualize limited number of feature map

	# if see_all_feature_map== True:
	# 	num_of_feature_maps= one_feature_map.shape[0]
	# 	num_of_rows= num_of_feature_maps/num_of_columns
	# 	print (num_of_rows)



	# for i in range(1,num_of_feature_maps+1):
		# plt.subplot(num_of_rows,num_of_columns,i)
	plt.imshow(misc.imresize(one_feature_map[feature_map_id],[256,256]),cmap='gray')
	plt.show()

def show_me_data (data):
	plt.figure()
	plt.subplot(1,3,1)
	plt.imshow(data[0],cmap='gray')
	plt.subplot(1,3,2)
	plt.imshow(data[1],cmap='gray')
	plt.subplot(1,3,3)
	plt.imshow(data[2],cmap='gray')
	plt.show()

show_me_feature_maps(outs)
# show_me_individual_feature_maps(outs)
# show_me_individual_feature_maps(outs,2)
# show_me_data(copy)