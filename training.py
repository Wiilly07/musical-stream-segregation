import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import setting

for i in range(setting.epoch):
	print(f'epoch: {i+1}')
	os.system('python training_util.py')