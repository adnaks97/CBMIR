import os 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
## defines how many slices you want to take pre and post the segmentation map given by TCIA
inp='/media/bmi/MIL/Nii_Converted_Classes_/phantom/'
output='/media/bmi/MIL/jnk_con/phantom'
if not os.path.exists(output):
    os.mkdir(output)

def saveArray(image,p,sc):

	image = image.get_data()
	x,y,z = np.where(image !=0) 
	z= np.unique(z)
	z= np.sort(z)
	print (z)
	cntr=0
	out_name = output + '/' + p
	if not os.path.exists(out_name):
	    os.mkdir(out_name)
	out_name = out_name + '/scan' + str(sc)
	if not os.path.exists(out_name):
	    os.mkdir(out_name)
	for slices in z:
	    im_slice = np.transpose(image[:,:,slices])
	    np.save(out_name + '/' + p + '_SCAN_' + str(sc) + '_SLICE_' + str(cntr) + '.npy',im_slice)
	    cntr=cntr + 1

def dcm2numpy(input):
	patients = os.listdir(input)
	for p in patients:
	    print 'patient id : ',p
	    imgs = os.listdir(input + p)
	    sc = 1
	    for i in imgs:
	    	image =nib.load(input + p + '/' + i)
	    	saveArray(image,p,sc)
	    	sc = sc + 1



if __name__ == '__main__':
	records = os.listdir(inp)
	for rec in records:
		dcm2numpy(inp + rec + '/')

