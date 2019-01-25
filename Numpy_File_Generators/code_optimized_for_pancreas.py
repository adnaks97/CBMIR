import os 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import re

slack = 1 ## defines how many slices you want to take pre and post the segmentation map given by TCIA
inp='/media/bmi/MIL/Nii_Converted_Classes_/Pancreas'
output='/media/bmi/MIL/numpyFiles/Pancreas'
label_home = '/media/bmi/MIL/Nii_Converted_Classes_/Pancreas/Pancreas_labels'

patient_map = dict()
label_map = dict()
if not os.path.exists(output):
    os.mkdir(output)

# mapping labels to numbers
for file in os.listdir(label_home):
    key = re.findall(r'\d+',file)[0]
    label_map[key] = os.path.join(label_home,file)

root, folders, files =os.walk(inp).next()
for fold in folders:
    key = fold.split('_')[1]
    if key == 'labels':
        continue    
    input = os.path.join(inp,fold)
    print input

    plist = os.listdir(input)
    patient_map[key] = os.path.join(input,plist[0])

for key in label_map.keys():
    patient = nib.load(patient_map[key])
    patient = patient.get_data()

    truth = nib.load(label_map[key])
    truth = truth.get_data()

    x,y,z = np.where(truth != 0)

    z = np.unique(z)

    z = np.sort(z)
    print (z)
    cntr=0

    for slices in z:
        #t2_slice =np.transpose(T2[:,:,slices])
        #t1_slice =np.transpose(T1[:,:,slices])
        pName = patient_map[key].split('/')[-2]
        out_name =os.path.join(output,pName)
        print out_name
        if not os.path.exists(out_name):
            os.mkdir(out_name)
        slice = np.rot90(patient[:,:,slices])
        np.save(os.path.join(out_name,'SLICE'+str(cntr)+'.npy'),slice)
        cntr=cntr+1

    



    # 
    # print plist
    # patients = list()

    # for p in plist:
    #     if '.mhd' in p and '_segmentation.mhd' not in p:
    #         patients.append(p)

    # fileLoc = input
    # for p in patients:
    #     scan = sitk.ReadImage(input + p)
    #     name = p.split('.mhd')[0]
    #     name = input + name + '_segmentation.mhd'
    #     mask = sitk.ReadImage(name)
    #     imgArray = sitk.GetArrayFromImage(scan)
    #     truth = sitk.GetArrayFromImage(mask)

