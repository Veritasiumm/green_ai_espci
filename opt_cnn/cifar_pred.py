
import numpy as np
from threading import Thread
from aux_fct_base import *
import gc, time, sys, glob

#Comment to access system wide install
#sys.path.insert(0,glob.glob('/content/CIANNA/src/build/lib.*/')[-1])
import CIANNA as cnn

load_epoch = 0

#Change image test mode in aux_fct to change network resolution in all functions
init_data_gen(test_mode=1)

#Batch size does not affect the inference result, but larger batch size process faster.
#Using FP16C_FP32A can sligtly reduce the accuracy, but it strongly accelerate computation.
cnn.init(in_dim=i_ar([image_size,image_size]), in_nb_ch=3, out_dim=nb_class,
	bias=0.1, b_size=16, comp_meth='C_CUDA', dynamic_load=1,
	mixed_precision="FP32C_FP32A", adv_size=30, inference_only=1)

#Compute on only half the validation set to reduce memory footprint
input_test, targets_test = create_val_batch()
cnn.create_dataset("TEST", nb_val, input_test[:,:], targets_test[:,:])

del (input_test, targets_test)
gc.collect()

if(load_epoch == 0):
	#If load epoch is 0, load the reference model instead
	if(not os.path.isfile("arch_ref_res32_err24.18_ms330.dat")):
		os.system("wget https://share.obspm.fr/s/TnLePm62SjCg4s4/download/arch_ref_res32_err24.18_ms330.dat")
	cnn.load("arch_ref_res32_err24.18_ms330.dat", load_epoch, bin=1)
else:
	cnn.load("net_save/net0_s%04d.dat"%load_epoch, load_epoch, bin=1)


cnn.forward(repeat=1, no_error=1, saving=2, drop_mode="AVG_MODEL")

start = time.perf_counter()
cnn.forward(no_error=1, saving=2, drop_mode="AVG_MODEL")
end = time.perf_counter()

cnn.perf_eval()

compute_time = (end-start)*1000 #in miliseconds
np.savetxt("compute_time.txt", [compute_time])

#Must be run with the following command for score eval to work: python3 cifar_pred_base.py 2>&1 | tee out.txt 
score_eval(load_epoch)

		
