import os 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import dicom

slack = 1 ## defines how many slices you want to take pre and post the segmentation map given by TCIA
inp='/media/bmi/MIL/Nii_Converted_Classes_/CBIS-DDSM_corrected'
output='/media/bmi/MIL/numpyFiles/CBIS-DDSM'

if not os.path.exists(output):
    os.mkdir(output)
root, folders, files =os.walk(inp).next()
for fold in folders:
    input = os.path.join(inp,fold)
    print input

    plist = os.listdir(input)
    try:
        file = plist[0]
    except:
        continue
    loc = os.path.join(input,file)
    pName = loc.split('/')[-2]
    out_name =os.path.join(output,pName)
    print out_name
    if not os.path.exists(out_name):
        os.mkdir(out_name)
    else:
        continue

    try:
    	patient = dicom.read_file(loc)
    	slice = patient.pixel_array
    	save_loc = os.path.join(out_name,'Array_SLICE.npy')
    	np.save(save_loc,slice)
    	print save_loc
    except:
    	print "error failed convert"
    	os.rmdir(out_name)
    	continue

