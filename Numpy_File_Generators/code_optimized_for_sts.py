import os
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as pyplot

home = '/media/bmi/MIL/Nii_Converted_Classes_/soft_tissue_sarcoma'
output='/media/bmi/MIL/numpyFiles/soft_tissue_sarcoma'

if not os.path.exists(output):
    os.mkdir(output)

for patient in os.listdir(home):
	path = os.path.join(home,patient)
	out_name = os.path.join(output,patient)
	if not os.path.exists(out_name):
		os.makedirs(out_name)
	else:
		continue
	scan = 0
	for file in os.listdir(path):
		patient_path = os.path.join(path,file)
		image = nib.load(patient_path)
		imageArray = image.get_data()

		x,y,z = np.where(imageArray != 0)

		z = np.unique(z)
		z = np.sort(z)
		print z

		cntr = 0
		for slices in z:
			slice = np.rot90(imageArray[:,:,slices])
			save_loc = os.path.join(out_name,'_SCAN_' + str(scan) + '_SLICE'+str(cntr)+'.npy')
			np.save(save_loc,slice)
			print save_loc
			cntr=cntr+1

		scan = scan + 1
		
