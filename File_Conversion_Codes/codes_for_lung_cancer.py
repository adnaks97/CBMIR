import os
import dicom2nifti

inp = '/media/bmi/MIL/24_datasets/LCTSC/LCTSC'
output= '/media/bmi/MIL/Nii_Converted_Classes_/Lung_Cancer'
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
        for seq2 in depth_2:
            # print seq2
            hnc_input_path= path_2+'/'+seq2
            # print hnc_input_path
            hnc_output_path = output+'/'+p
            if not os.path.exists(hnc_output_path):
                os.mkdir(hnc_output_path)
            try:
                dicom2nifti.dicom_series_to_nifti(hnc_input_path, hnc_output_path+'/'+p+'.nii.gz', reorient_nifti=True)
            except:
                print
            # dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=True)
            # os.system('dicom2nifti '+hnc_input_path+' '+hnc_output_path)                
        print '***'
