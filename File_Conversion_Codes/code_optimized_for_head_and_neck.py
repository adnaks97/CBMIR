import os
inp = '/media/bmi/MIL/jnk/hncs'
output= '/media/bmi/MIL/nii_converted_classes/Head_and_Neck/hncs'

files, patients, folders = os.walk(inp).next()

for p in patients:
	path    = files+'/'+p
	depth_1 = os.listdir(path)
	for seq in depth_1:
		print seq
		path_2 = path+'/'+seq
		depth_2 =os.listdir(path_2)
		for seq2 in depth_2:
			# print seq2
			hnc_input_path= path_2+'/'+seq2
			print hnc_input_path
			hnc_output_path = output+'/'+p
			if not os.path.exists(hnc_output_path):
				os.mkdir(hnc_output_path)
			os.system('dicom2nifti '+hnc_input_path+' '+hnc_output_path)				
		print '***'
