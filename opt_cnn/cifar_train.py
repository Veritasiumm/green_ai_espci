	
import numpy as np
from threading import Thread
from aux_fct import *
import gc, sys, glob

#Comment to access system wide install
#sys.path.insert(0,glob.glob('/content/CIANNA/src/build/lib.*/')[-1])
import CIANNA as cnn


def data_augm():
	input_data, targets = create_train_batch()
	cnn.delete_dataset("TRAIN_buf", silent=1)
	cnn.create_dataset("TRAIN_buf", nb_images_per_iter, input_data[:,:], targets[:,:], silent=1)
	return


total_iter = 900 #Larger architecture are likely to require more iterations
nb_iter_per_augm = 1 #Increase if augmentation is slower that training on one iteration
if(nb_iter_per_augm > 1):
	shuffle_frequency = 1
else:
	shuffle_frequency = 0

load_iter = 600 #Used to select the net_save file from which the training must be restarted

start_iter = int(load_iter / nb_iter_per_augm)

#The batch size can be adapted to reduce the RAM footprint, but it is likely to affect the reachable accuracy
#The mixed precision can be switched to FP16C_FP32A to speed up training on T4 GPUs, 
#but it can induce vanishing or exploding gradients
cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=nb_class,
		bias=0.1, b_size=16, comp_meth='C_CUDA', dynamic_load=1,
		mixed_precision="FP32C_FP32A", adv_size=30)

init_data_gen()

input_val, targets_val = create_val_batch()
cnn.create_dataset("VALID", nb_val, input_val[:,:], targets_val[:,:])
cnn.create_dataset("TEST", nb_val, input_val[:,:], targets_val[:,:])
del (input_val, targets_val)
gc.collect()

input_data, targets = create_train_batch()
cnn.create_dataset("TRAIN", nb_images_per_iter, input_data[:,:], targets[:,:])

if(load_iter > 0):
	#Load a model save file based on the default naming scheme
	cnn.load("net_save/net0_s%04d.dat"%load_iter, load_iter, bin=1)
	#You can change the file path to load a renamed saved model; e.g.:
	#cnn.load("my_model.dat", load_iter, bin=1)
else:
	#Create a new network structure to train from scratch
	#See the CIANNA API description for a list of available layers and there parameters
	cnn.conv(f_size=i_ar([5,5]), nb_filters=8 , padding=i_ar([2,2]), activation="RELU")
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.conv(f_size=i_ar([5,5]), nb_filters=16, padding=i_ar([2,2]), activation="RELU")
	cnn.pool(p_size=i_ar([2,2]), p_type="MAX")
	cnn.dense(nb_neurons=256, activation="RELU", drop_rate=0.5)
	cnn.dense(nb_neurons=128, activation="RELU", drop_rate=0.2)
	cnn.dense(nb_neurons=nb_class, strict_size=1, activation="SMAX")

for run_iter in range(start_iter,int(total_iter/nb_iter_per_augm)):

	t = Thread(target=data_augm)
	t.start()

	#See the CIANNA API description for a list and descroption of available keywords
	cnn.train(nb_iter=nb_iter_per_augm, learning_rate=0.002, end_learning_rate=0.00003, shuffle_every=shuffle_frequency ,\
			 control_interv=20, confmat=1, momentum=0.9, lr_decay=0.0015, weight_decay=0.0005, save_every=20,\
			 silent=0, save_bin=1, TC_scale_factor=256.0)

	if(run_iter == start_iter):
		cnn.perf_eval()

	t.join()
	cnn.swap_data_buffers("TRAIN")

	

















