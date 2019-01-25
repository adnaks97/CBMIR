import os 
import nrrd
import numpy as np
import matplotlib.pyplot as plt

inp='/media/bmi/MIL/Mandible/'
output='/media/bmi/MIL/jnk_con/MANDIBLE/'
file = '/structures/Mandible.nrrd'

def getTruth(patient):
	folder =  patient + '/structures/'
	lst = os.listdir(folder)
	files = list()
	for file in lst:
		if 'Submandibular' in file:
			files.append(file)
	print files
	if len(files) == 0:
		return np.arange(0)
	univTruth = np.zeros(((nrrd.read(folder + '/' + files[0]))[0]).shape,dtype='uint8')
	print univTruth.shape
	for file in files:
		mask,options = nrrd.read(folder + '/' + file)
		print mask.shape
		univTruth[np.where(mask != 0)] = 1
	return univTruth


if not os.path.exists(output):
	os.mkdir(output)

root,folders,junk = os.walk(inp).next()
for folder in folders:
	if folder in ['PDDCA-1.4.1_part1','PDDCA-1.4.1_part3']:  ##comment this entire if block to run code afresh
		continue
	current,serialList,junk = os.walk(inp +folder + '/').next()
	print current
	for serial in serialList:
		patient,Structures,files = os.walk(current + serial).next()
		if os.path.exists(output + serial): #comment to re-write existing files
			continue
		print patient
		segFile = patient + file
		if os.path.exists(segFile):
			segment,option1 = nrrd.read(segFile)
		else:
			segment = getTruth(patient)
		if segment.size == 0:
			print "check"
			continue
		mainFile,option2 = nrrd.read(patient + '/img.nrrd')

		x,y,z = np.where(segment != 0)
		z = np.unique(z)
		z = np.sort(z)
		print (z)
		cntr=0

		for slices in z:
			out_name =output + serial + '/'
			#print out_name
			if not os.path.exists(out_name):
				os.mkdir(out_name)
			mainSlice = np.transpose(mainFile[:,:,slices])
			#plt.imshow(mainSlice,cmap = 'gray')
			#plt.show()
			np.save(out_name  + 'SLICE'+str(cntr)+'.npy',mainSlice)
			cntr=cntr+1


