import os

home = '/media/bmi/MIL/24_datasets/CBIS-DDSM/CBIS-DDSM'
new_home = '/media/bmi/MIL/Nii_Converted_Classes_/CBIS-DDSM_corrected'
os.makedirs(new_home)

for patient in os.listdir(home):
	depth1 = os.path.join(home,patient)
	for seq1 in os.listdir(depth1):
		depth2= os.path.join(depth1,seq1)
		for seq2 in os.listdir(depth2):
			path = os.path.join(depth2,seq2)
			if len(os.listdir(path)) == 1:
				new_patient = os.path.join(new_home,patient)
				try:
				    os.makedirs(new_patient)
				    break
				except OSError, e:
				    if e.errno != os.errno.EEXIST:
				        raise   
				    # time.sleep might help here
				    pass				
				for file in os.listdir(path):
					print "Moving : " + patient
					os.system("cp " + os.path.join(path,file) + " " + new_patient)