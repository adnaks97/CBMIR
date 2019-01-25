import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline

def sidebyside(a,b):
    fig = plt.figure()
    x=fig.add_subplot(1,2,1)
    plt.imshow(a,cmap = 'gray')
    x.set_title("original")
    x=fig.add_subplot(1,2,2)
    plt.imshow(b,cmap = 'gray')
    x.set_title("flipped")
    plt.show()
    
save_home = '/media/bmi/MIL/Flipped'
numpy_home = '/media/bmi/MIL/numpyFiles_resized'
anatomy_list = ['Kidney','Thyroid','stomach','ovarian','Liver','uterus','esophagus','Rectum','Colon']
for anatomy in anatomy_list:
    organ_flip = os.path.join(save_home,anatomy)
    if not os.path.exists(organ_flip):
        os.makedirs(organ_flip)
    else:
        continue
    organ_home = os.path.join(numpy_home,anatomy + '_resized')
    for patient in os.listdir(organ_home):
        patient_home = os.path.join(organ_home,patient)
        patient_flip = os.path.join(organ_flip,patient)
        if not os.path.exists(patient_flip):
            os.makedirs(patient_flip)
        for image in os.listdir(patient_home):
            loc = os.path.join(patient_home,image)
            a = np.load(loc)
            flipped = np.flip(a,1)
            name = image.split('.')[0] + '_FLIPPED.npy'
            np.save(os.path.join(patient_home,name),flipped)
            sidebyside(a,flipped)
