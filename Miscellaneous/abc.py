import time,os
time_str =  time.asctime(time.localtime(time.time())).split()[3]
vals = time_str.split(':')
while (int(vals[0]) >= 18 and int(vals[1]) > 30):
	time_str =  time.asctime(time.localtime(time.time())).split()[3]
	vals = time_str.split(':')
os.system("python feature_Extraction.py") 