import os 
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

slack = 1 ## defines how many slices you want to take pre and post the segmentation map given by TCIA
inp='/media/bmi/MIL/promise-12/'
output='/media/bmi/MIL/numpyFiles/PROSTRATE/'
if not os.path.exists(output):
    os.mkdir(output)
root, folders, files =os.walk(inp).next()
for fold in folders:
    input = inp + fold + '/'
    print input

    plist = os.listdir(input)
    print plist
    patients = list()

    for p in plist:
        if '.mhd' in p and '_segmentation.mhd' not in p:
            patients.append(p)

    fileLoc = input
    for p in patients:
        scan = sitk.ReadImage(input + p)
        name = p.split('.mhd')[0]
        name = input + name + '_segmentation.mhd'
        mask = sitk.ReadImage(name)
        imgArray = sitk.GetArrayFromImage(scan)
        truth = sitk.GetArrayFromImage(mask)

        x,y,z = np.where(truth != 0)

        x = np.unique(x)

        x = np.sort(x)
        print (x)
        cntr=0

        for slices in x:
            #t2_slice =np.transpose(T2[:,:,slices])
            #t1_slice =np.transpose(T1[:,:,slices])
            pName = p.split('.mhd')[0]
            out_name =output + pName + '/'
            print out_name
            if not os.path.exists(out_name):
                os.mkdir(out_name)
            np.save(out_name  + 'SLICE'+str(cntr)+'.npy',imgArray[slices,:,:])
            cntr=cntr+1