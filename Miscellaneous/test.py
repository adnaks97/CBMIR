import os
inp = '/media/bmi/MIL/24_datasets/phantom'
output= '/media/bmi/MIL/jnk_con/phantom'

if not os.path.exists(output):
    os.mkdir(output)

files, patients, folders = os.walk(inp).next()

for p in patients:
    path    = files+'/'+p
    depth_1 = os.listdir(path)
    for seq in depth_1:
        print seq
        path_2 = path+'/'+seq
        depth_2 =os.listdir(path_2)
        for seq2 in depth_2:
            path_3 = path + seq2
            depth_3 = os.listdir(path_3)
            # print seq2
            for fold in depth_3:
                phantom_input_path= path_3+'/'+fold
                print phantom_input_path
                phantom_output_path = output+'/'+p + '/' + seq
                if not os.path.exists(phantom_output_path):
                    os.mkdir(phantom_output_path)
                dicom2nifti.convert_directory(phantom_input_path, phantom_output_path)                
            print '***'





