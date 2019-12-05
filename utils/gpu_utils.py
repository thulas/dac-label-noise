import subprocess
import numpy as np
import socket
#import pdb

def get_num_gpus():
	nvo = subprocess.Popen(['nvidia-smi','-L'],stdout=subprocess.PIPE)
	num_devices = subprocess.check_output(('wc', '-l'),
	                               stdin=nvo.stdout).strip()
	try:
		return int(num_devices)
	except ValueError:
		return 0



# p1 = subprocess.Popen(['convert', fileIn, 'bmp:-'], stdout=subprocess.PIPE)
# p2 = subprocess.Popen(['mkbitmap', '-f', '2', '-s', '2', '-t', '0.48'], 
#      stdin=p1.stdout, stdout=subprocess.PIPE)
# p1.stdout.close()
# p3 = subprocess.Popen(['potrace', '-t' , '5', '-s' , '-o', fileOut],        
#      stdin=p2.stdout,stdout=subprocess.PIPE)
# p2.stdout.close()

# output = p3.communicate()[0]

"""
def get_free_gpu_list(gpu_range):
	num_gpus=get_num_gpus()
	free_gpus = []
	import random, time
	#if True:
	#sleep for a random time between 0 and 2 seconds to break symmetry
	#between this process and others on the same compute node. otherwise
	#processes end up accessing the same gpu.
	sleep_time = np.random.uniform(0,num_gpus)
	wid = socket.gethostname()
	print "sleeping for time %f on %s" %(sleep_time,wid)
	time.sleep(sleep_time)

	for gpu_id in range(gpu_range):
	    nvo = subprocess.Popen(['nvidia-smi','-q','-i',str(gpu_id),],stdout=subprocess.PIPE)
	    # busy = subprocess.check_output(('grep', 'Processes'),
	    #                                stdin=nvo.stdout).strip().split()[-1]
	    #nvo_process = subprocess.Popen(['grep', '"Process ID"'],
	    #                                stdin=nvo.stdout,stdout=subprocess.PIPE)
	    #nvo.stdout.close()


	    if busy == 'None':
	    	free_gpus.append(gpu_id)
	return free_gpus

"""

import pdb

#a better way to get free GPU list
def get_free_gpu_list(gpu_range):
	num_gpus=get_num_gpus()
	free_gpus = []
	import random, time
	#if True:
	#sleep for a random time between 0 and 2 seconds to break symmetry
	#between this process and others on the same compute node. otherwise
	#processes end up accessing the same gpu.
	#sleep_time = np.random.uniform(0,num_gpus)
	sleep_time = np.random.uniform(0,4)
	wid = socket.gethostname()
	print("sleeping for time %f on %s" %(sleep_time,wid))
	time.sleep(sleep_time)

	#pdb.set_trace()
	for gpu_id in range(gpu_range):
	    #nvo = subprocess.Popen(['nvidia-smi','-q','-i',str(gpu_id),],stdout=subprocess.PIPE)
	    gpu_str_id = '--id='+str(gpu_id)
	    nvo = subprocess.Popen(['nvidia-smi','--query-gpu=utilization.gpu',
	     gpu_str_id, '--format=csv'],stdout=subprocess.PIPE)
	    #awk_str = "awk '{print $1}' "
	    awko = subprocess.Popen(['awk',"{{print $1}}"],stdin=nvo.stdout,stdout=subprocess.PIPE)
	    #nvo.stdout.close()
	    busy = subprocess.check_output(('grep','-v', 'utilization'),
	    	stdin=awko.stdout).rstrip()
	    awko.stdout.close()
	    #busy = subprocess.check_output(('grep', 'Processes'),
	    #                                stdin=nvo.stdout).strip().split()[-1]
	    #nvo_process = subprocess.Popen(['grep', '"Process ID"'],
	    #                                stdin=nvo.stdout,stdout=subprocess.PIPE)
	    #nvo.stdout.close()
	    if busy == '0':
	    	free_gpus.append(gpu_id)
	return free_gpus



"""
def get_free_gpu(gpu_range):
	\"""                                                                                                                                                                                                
	returns first non-busy GPU in range(gpu_range)                                                                                                                                                     
	\"""
	for gpu_id in range(gpu_range):
	    nvo = subprocess.Popen(['nvidia-smi','-q','-i',str(gpu_id),],stdout=subprocess.PIPE)
	    busy = subprocess.check_output(('grep', 'Processes'),
	                                   stdin=nvo.stdout).strip().split()[-1]
	    if busy == 'None':
	        return gpu_id
	return None
"""

#a better way to get free_gpu
def get_free_gpu(gpu_range):
	"""                                                                                                                                                                                                
	returns first non-busy GPU in range(gpu_range)                                                                                                                                                     
	"""
	#pdb.set_trace()
	for gpu_id in range(gpu_range):
	    #nvo = subprocess.Popen(['nvidia-smi','-q','-i',str(gpu_id),],stdout=subprocess.PIPE)
	    gpu_str_id = '--id='+str(gpu_id)
	    nvo = subprocess.Popen(['nvidia-smi','--query-gpu=utilization.gpu',
	     gpu_str_id, '--format=csv'],stdout=subprocess.PIPE)
	    #awk_str = "awk '{print $1}' "
	    awko = subprocess.Popen(['awk',"{{print $1}}"],stdin=nvo.stdout,stdout=subprocess.PIPE)
	    #nvo.stdout.close()
	    busy = subprocess.check_output(('grep','-v', 'utilization'),
	    	stdin=awko.stdout).rstrip()
	    awko.stdout.close()
	    #busy = subprocess.check_output(('grep', 'Processes'),
	    #                                stdin=nvo.stdout).strip().split()[-1]
	    #nvo_process = subprocess.Popen(['grep', '"Process ID"'],
	    #                                stdin=nvo.stdout,stdout=subprocess.PIPE)
	    #nvo.stdout.close()
	    if type(busy)==bytes:
	    	busy = busy.decode("utf-8")
	    if busy == '0':
	        return gpu_id
	return None



def get_cuda_device(args):
	#todo: this should be determined automatically
	#num_gpus=2
	num_gpus=get_num_gpus()

	if args.use_gpu:
		if args.cuda_device == 'auto':
			print("trying to auto-detect free GPU")
			#to do: move the function to another module, since this
			#does not have anything to do with theano
			import random, time
			#if True:
			#sleep for a random time between 0 and 2 seconds to break symmetry
			#between this process and others on the same compute node. otherwise
			#processes end up accessing the same gpu.
			#sleep_time = np.random.uniform(0,num_gpus)
			sleep_time = np.random.uniform(0,4)
			wid = socket.gethostname()
			print("sleeping for time %f on %s" %(sleep_time,wid))
			time.sleep(sleep_time)
			gpu = get_free_gpu(num_gpus)
			if not (gpu is None):
				cuda_device_id = int(gpu)
			else:
				cuda_device_id = None

		else: # use user specified cuda device
			cuda_device_id = int(args.cuda_device)
		print("using GPU ", cuda_device_id)

	else:
		cuda_device_id = None

	return cuda_device_id
