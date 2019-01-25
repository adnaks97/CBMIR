import os 
import numpy as np
import scipy
from scipy import misc
from scipy.misc import imread 
import matplotlib.pyplot as plt 
# import 
input ='/media/bmi/MIL/24_datasets/EYE_RETINA/resources/images/ddb1_fundusimages'
output=''
if  not os.path.exists(output):
	os.mkdir(output)

images= os.listdir(input)
cntr=0
for i in images :
	print i
	# if cntr > 1:
	# 	break
	eye = imread(input+'/'+i)
	r_eye = eye[:,:,0]
	g_eye = eye[:,:,1]
	b_eye = eye[:,:,2]
	name = i.split('.png')
	name =name[0]
	out_folder = output+'/'+name
	if not os.path.exists(out_folder):
		os.mkdir(out_folder)

	# plt.imshow(r_eye,cmap='gray')
	# plt.figure()
	# plt.imshow(b_eye,cmap='gray')
	# plt.figure()
	# plt.imshow(g_eye,cmap='gray')
	# plt.figure()
	np.save(out_folder+'/'+name+'_red.npy',r_eye)
	np.save(out_folder+'/'+name+'_green.npy',g_eye)
	np.save(out_folder+'/'+name+'_blue.npy',b_eye)		
	cntr= cntr+1
print ('number of images =',cntr)
	#plt.figure()
# plt.show()
