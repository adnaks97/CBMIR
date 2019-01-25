import os 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

slack = 1 ## defines how many slices you want to take pre and post the segmentation map given by TCIA
input='/media/bmi/MIL/LGG/'
output='/media/bmi/MIL/jnk_con/BRAIN'
if not os.path.exists(output):
    os.mkdir(output)
patients = os.listdir(input)

for p in patients:
    print 'patient id : ',p
    imgs = os.listdir(input+'/'+p)

    for i in imgs:

        if 'seg' in i:
            truth =nib.load(input + p + '/' + i)
        if 't2.' in i:
            T2 =nib.load(input + p + '/' + i)
        if 't1.' in i:
            T1 =nib.load(input + p + '/' + i)

    truth= truth.get_data()
    T2   = T2.get_data()
    T1   = T1.get_data()

    x,y,z = np.where(truth !=0) 
    z= np.unique(z)
    """# print z
    for sl in range(-slack,slack+1):
        # print 'sl',sl
        if sl < 0 :
            z= np.append(z,np.min(z)+sl)
        if sl > 0:
            z= np.append(z,np.max(z)+sl)
        # if sl ==0:
            # print""" 
    z= np.sort(z)
    print (z)
    cntr=0
    for slices in z:
        t2_slice =np.transpose(T2[:,:,slices])
        t1_slice =np.transpose(T1[:,:,slices])
        out_name =output+'/'+p
        if not os.path.exists(out_name):
            os.mkdir(out_name)
        np.save(out_name+'/'+ p + '_' + str(cntr) + '.npy',t2_slice)
        cntr=cntr+1
        np.save(out_name+'/'+ p + '_' + str(cntr) + '.npy',t1_slice)
        cntr=cntr+1






#       plt.figure()
#       plt.imshow(sli,cmap='gray')
# plt.show()