import os
import dicom2nifti
from pathlib import Path

input = '/media/bmi/MIL/24_datasets/'
output= '/media/bmi/MIL/Nii_Converted_Classes_'


def convert(inp):
	files, patients, folders = os.walk(inp).next()

	for p in patients:
		path    = files+'/'+p
		depth_1 = os.listdir(path)
		phantom_output_path = output+'/'+p
		if not os.path.exists( phantom_output_path ):
			os.mkdir( phantom_output_path )
		else:
			continue
		for seq in depth_1:
			#print seq
			path_2 = path+'/'+seq
			depth_2 =os.listdir(path_2)
			for seq2 in depth_2:
				#print seq2
				myFile = Path(phantom_output_path + '/' + p + '_' + seq + '_' + seq2 + '.nii.gz')
				if myFile.is_file():
					continue
				print
				phantom_input_path= path_2+'/'+seq2
				# print phantom_input_path
				try:
					print 'Converting --- ',p + '_' + seq + '_' + seq2
					dicom2nifti.dicom_series_to_nifti(phantom_input_path, phantom_output_path + '/' + p + '_' + seq + '_' + seq2 + '.nii.gz', reorient_nifti=True)
					print '***DONE***'
				except:
					continue
				#dicom2nifti.convert_directory(phantom_input_path, phantom_output_path)
				# os.system('dicom2nifti '+phantom_input_path+' '+phantom_output_path)	
				#print phantom_input_path,"|||",phantom_output_path				
			print '___________________________________________________________________________________________________'
		print '-------------------------------------------------------------------------------------------------------'

if __name__ == '__main__':

	fin = raw_input('Enter Input Folder (Only Folder Name)')
	out = raw_input('Enter output Folder (Only Folder Name)')
	input = input + fin
	output = output + out
	if not os.path.exists(output):
		os.mkdir(output)
	types = os.listdir(input)
	for type in types :
		inp = input + '/' + type
		convert(inp)
	
