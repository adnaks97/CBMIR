import random,os,matplotlib.pyplot as plt,numpy as np
home = '/media/bmi/MIL/23_class_net/train'
k = 1
for anatomy in os.listdir(home):
#     if k>5 :
#         break
    anatomy_path = os.path.join(home,anatomy)
    patient_list = os.listdir(anatomy_path)
    i = random.randint(0,len(patient_list)-1)
#     print i, patient_list[i]
    patient_path = os.path.join(anatomy_path,patient_list[i])
    file_list = os.listdir(patient_path)
    j = random.randint(0,len(file_list)-1)
#     print j, file_list[j]
    file_path = os.path.join(patient_path,file_list[j])
    arr = np.load(file_path)
    plt.subplot(5,5,k)
    plt.imshow(arr,cmap = 'gray')
    plt.title(anatomy)
    plt.axis("off")
    k = k+1
plt.subplots_adjust(hspace = 0.9,wspace = 0.9)
plt.savefig("AllClasses.png")
plt.show()