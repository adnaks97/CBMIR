import numpy as np 
import nibabel as nib
import os 
import matplotlib.pyplot as plt
input_path ='/media/bmi/MIL/24_datasets/LIVER/LiTs2017/Training/media'
output_path='/media/bmi/MIL/Nii_Converted_Classes_/Liver'

slack =8
segmentation_files=[]
input_image=[]
images= os.listdir(input_path)
for i in images:
    if 'volume' in i:
        input_image.append(i)
    if 'segmentation' in i:
        segmentation_files.append(i)

input_image.sort()
segmentation_files.sort()

for i in xrange(len(input_image)):

    name = input_image[i]
    name =name.split('.nii')
    name =name[0]
    print input_image[i]
    if not os.path.exists(output_path+'/'+name):
        os.mkdir(output_path+'/'+name)

    volume = nib.load(input_path+'/'+input_image[i]).get_data()
    print (input_image[i])
    segmentation= nib.load(input_path+'/'+segmentation_files[i]).get_data()
    print (segmentation_files[i])
    x,y,z = np.where(segmentation==1)
    dim=[]
    slice_iterator=[]
    for slices in np.unique(z):
        x,y  = np.where(np.rot90(segmentation[:,:,int(slices)])==1)
        liver_count = len(x)
        dim = np.append(dim,liver_count)
        slice_iterator=np.append(slice_iterator,slices)


    slice_of_interest=slice_iterator[np.where(dim==np.max(dim))]
    print slice_of_interest
    mini =np.min(slice_of_interest)
    maxi =np.max(slice_of_interest)
    for s in range (-slack,slack):
        if s <0:
            slice_of_interest=np.append(slice_of_interest,mini+s)
        if s >0:
            slice_of_interest=np.append(slice_of_interest,maxi+s)          
    slice_of_interest=np.sort(slice_of_interest)
    # print slice_of_interest
    cntr=0    
    for slices in slice_of_interest:
        # print slices

        one_slice = np.rot90(volume[:,:,int(slices)])
        np.save(output_path+'/'+name+'/'+name+'_'+str(cntr)+'.npy',one_slice)
        cntr=cntr+1


#         plt.figure()
#         plt. imshow(one_slice,cmap='gray')
#         cntr=cntr+1
#         if cntr>10:
#             break
# plt.show()


