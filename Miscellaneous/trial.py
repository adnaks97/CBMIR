import os
import dicom2nifti

inp = '/media/bmi/MIL/24_datasets/esophagus/TCGA-ESCA'
output= '/media/bmi/MIL/jnk_con/eso'
if not os.path.exists(output):
    os.mkdir(output)

files, patients, folders = os.walk(inp).next()

for p in patients:
    print p
    path    = files+'/'+p
    depth_1 = os.listdir(path)
    for seq in depth_1:
        # print seq
        path_2 = path+'/'+seq
        depth_2 =os.listdir(path_2)
        dicom_directory =path_2
        output_folder= output+'/'+p 
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        dicom2nifti.convert_directory(dicom_directory, output_folder)
