import os 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import re

slack = 1 ## defines how many slices you want to take pre and post the segmentation map given by TCIA
inp='/media/bmi/MIL/Nii_Converted_Classes_/lymph'
output='/media/bmi/MIL/numpyFiles/lymph'
#label_home = '/media/bmi/MIL/Nii_Converted_Classes_/Pancreas/Pancreas_labels'

# patient_map = dict()
# label_map = dict()
if not os.path.exists(output):
    os.mkdir(output)

# mapping labels to numbers
# for file in os.listdir(label_home):
#     key = re.findall(r'\d+',file)[0]
#     label_map[key] = os.path.join(label_home,file)
Error_List = list()
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
    patient = nib.load(loc)
    patient = patient.get_data()

    z = np.arange(540,560)
    print (z)
    cntr=0

    if np.max(z) > patient.shape[2]:
        print 540,560
        print patient.shape
        print "Error cannot process file - out of range"
        Error_List.append(loc) 
        print "*************************************************"
        os.rmdir(out_name)
        continue

    for slices in z:
        #t2_slice =np.transpose(T2[:,:,slices])
        #t1_slice =np.transpose(T1[:,:,slices])
        try:
            slice = np.rot90(patient[:,:,slices])
            save_loc = os.path.join(out_name,'SLICE'+str(cntr)+'.npy')
            np.save(save_loc,slice)
            print save_loc
            cntr=cntr+1
        except:
            print "slice out of bounds"
            continue





