import shutil
import pathlib

def clean(path):
	parts = path.split('_')
	path1 = '/media/bmi/MIL/24_datasets/renal/TCGA-KIRC'
	path2 = '/media/bmi/MIL/jnk_con/renal'
	for part in parts:
		path1 = path1 + '/' + part
	path2 = path2 + '/' + parts[0]
	shutil.rmtree(path1)
	shutil.rmtree(path2)