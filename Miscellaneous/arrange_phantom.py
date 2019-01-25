import os
root  = "/media/bmi/MIL/numpyFiles/phantom"
for folder in (os.listdir(root)):
	if folder in ['p0t0','p1t1','arrange.py']:
		continue
	patient = os.path.join(root,folder)
	for scan in (os.listdir(patient)):
		scanFolder = os.path.join(patient,scan)
		for file in os.listdir(os.path.join(patient,scan)):
			os.system("mv " + os.path.join(scanFolder,file) + " " + patient )
		os.rmdir(scanFolder)