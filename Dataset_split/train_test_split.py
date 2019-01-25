import os
import numpy as np 

def splitList(path,organ):
	patients = os.listdir(path)
	n_train = int(round(0.7 * len(patients)))
	n_test = int(round(0.1 * len(patients)))
	n_val = int(round(0.2 * len(patients)))

	print "check"

	train = patients[:n_train]
	test = patients[n_train:n_train + n_test]
	validate = patients[n_test + n_train:]

	# MODIFY DATASET SOURCE LOCATION IF NEEDED
	train_home = os.path.join("/media/bmi/MIL/23_class_net/train",organ)
	val_home = os.path.join("/media/bmi/MIL/23_class_net/val",organ)
	test_home = os.path.join("/media/bmi/MIL/23_class_net/test",organ)

	# organ = os.path.join("/media/bmi/MIL/6_class_net/",organ)
	
	# save_v = os.path.join(organ,"Validation_set")
	# save_tr = os.path.join(organ,"Training_set")
	# save_t = os.path.join(organ,"Test_set")

	if not os.path.exists(train_home):
		os.makedirs(train_home)
		print "\nTRAINING\n"
		for tr in train:
			os.system("cp -r " + os.path.join(path,tr) + " " + train_home)
			print "Copied : " + os.path.join(path,tr)

	if not os.path.exists(val_home):
		os.makedirs(val_home)
		print "\nVALIDATION\n"
		for val in validate:
			os.system("cp -r " + os.path.join(path,val) + " " + val_home)
			print "Copied : " + os.path.join(path,val)

	if not os.path.exists(test_home):
		os.makedirs(test_home)
		print "\nTEST\n"
		for t in test:
			os.system("cp -r " + os.path.join(path,t) + " " + test_home)
			print "Copied : " + os.path.join(path,t)

	print "check"

	# 	os.makedirs(save_v)
	# 	os.makedirs(save_t)
	# 	os.makedirs(save_tr)
	# else:
	# 	return
	# print save_v
	# print save_tr
	# print save_t

	print n_val,n_train,n_test

if __name__ == "__main__":

	print "check"

	root = "/media/bmi/MIL/numpyFiles_resized"
	print "check"
	for fold in os.listdir(root):
		splitList(os.path.join(root,fold), fold.split('_')[0] )
