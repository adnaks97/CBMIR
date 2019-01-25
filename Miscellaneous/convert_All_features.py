import os,numpy as np,matplotlib.pyplot as plt
%matplotlib inline
home = '/media/bmi/MIL/codes/numpyFeatures_23_class'
file_list = os.listdir(home)
total_rep = np.zeros((1,512))
total_patients = []
for file in file_list:
    arr = np.load(os.path.join(home,file))
#     print arr.shape,total_patients.shape
    if 'patient' in file:
        total_patients =np.append(total_patients,arr)
    else:
        total_rep = np.concatenate((total_rep,arr),axis = 0)
#x = np.delete(total_patients,total_patients[0],axis = 0)
y = np.delete(total_rep,total_rep[0],axis = 0)
print 
print len(total_patients),y.shape
total_patients = np.asarray(total_patients)
np.save(os.path.join(home,"total_patients.npy"),total_patients)
np.save(os.path.join(home,"total_rep.npy"),y)