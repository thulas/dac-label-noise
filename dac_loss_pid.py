"""
loss function definitions for deep abstaining classifier.
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import pdb
import math

from pid import pid_controller as PID

#for numerical stability
epsilon = 1e-7


def get_abst_rate(p_out):
	"""
	function to return abstention rate given a batch of outputs.
	p_out: a batch  of probabilisites over classes (or pre-softmax scores)
	returns: the rate of abstention; abstention class is assumed to be final class.
	"""
	#pdb.set_trace()
	abst_class_id = p_out.shape[1] - 1
	predictions = torch.argmax(p_out,dim=1)
	num_abstains = torch.sum(predictions.eq(abst_class_id))
	return torch.sum(num_abstains.float())/p_out.shape[0]


class dac_loss_pid(_Loss):
	def __init__(self, model, learn_epochs, total_epochs, use_cuda=False, cuda_device=None, 
		abst_rate=0.1, final_abst_rate=None, alpha_final=1.0,alpha_init_factor=64.,pid_tunings=(1.,0.,0.)):
		print("using dac-pid loss function\n")
		super(dac_loss_pid, self).__init__()
		self.model = model
		self.learn_epochs = learn_epochs
		self.total_epochs = total_epochs
		self.alpha_final = alpha_final
		self.alpha_init_factor = alpha_init_factor
		self.use_cuda = use_cuda
		self.cuda_device = cuda_device
		self.alpha_var = None
		self.alpha_thresh_ewma = None   #exponentially weighted moving average for alpha_thresh
		self.alpha_thresh = None #instantaneous alpha_thresh
		self.ewma_mu = 0.05 #mu parameter for EWMA; 
		self.curr_alpha_factor  = None #for alpha initiliazation
		self.alpha_inc = None #linear increase factor of alpha during abstention phase
		self.alpha_check_epoch = None
		self.abst_rate = None #instantaneous abstention rate
		self.abst_rate_ewma = None # exponentially weighted moving averge of abstention
		self.pid = None
		self.final_abst_rate = None

		if abst_rate < 0. or abst_rate is None:
			raise ValueError("Invalid abstention rate")


		k_p, k_i, k_d = pid_tunings
		
		self.pid = PID(set_point=abst_rate, k_p=k_p,
			k_i=k_i, k_d=k_d, limits=(-1.,1.))

		if final_abst_rate is not None:
			self.abst_delta = (abst_rate - final_abst_rate)/(self.total_epochs-self.learn_epochs)
			self.final_abst_rate = final_abst_rate


	def update_pid_abst_set_point(self, abst_delta):
		#new_abst_rate  = max(0.,self.pid.setpoint - abst_delta)
		new_abst_rate  = max(0.,self.pid.set_point - abst_delta)
		#self.pid.setpoint = new_abst_rate
		self.pid.set_point = new_abst_rate
		print("DAC updated abstention rate to %f" %(new_abst_rate))


	#update both instantaneous and ewma for alpha threhold
	def update_alpha_thresh(self, h_c, p_out_abstain):
		self.alpha_thresh = ((1. - p_out_abstain)*h_c).mean().detach().clone()
		#update alpha_thresh_ewma 
		if self.alpha_thresh_ewma is None:
			self.alpha_thresh_ewma = self.alpha_thresh
		else:
			self.alpha_thresh_ewma = (self.ewma_mu*self.alpha_thresh + \
							(1. - self.ewma_mu)*self.alpha_thresh_ewma).detach().clone()


	#update both instantaneous and ewma for abstention rate
	def update_abst_rate(self, p_out):
		self.abst_rate = get_abst_rate(p_out)
		if self.abst_rate_ewma is None:
			self.abst_rate_ewma = self.abst_rate
		else:
			self.abst_rate_ewma = self.ewma_mu*self.abst_rate + \
			(1-self.ewma_mu)*self.abst_rate_ewma


	def get_loss_terms(self,input_batch, target_batch):
		#calculate cross entropy only over true classes
		h_c = F.cross_entropy(input_batch[:,0:-1],target_batch,reduction='none')
		p_out = torch.exp(F.log_softmax(input_batch,dim=1))
		p_out_abstain = p_out[:,-1]
		# avoid numerical instability by upper-bounding 
		# p_out_abstain to never be more than  1 - eps since we have to
		# take log(1 - p_out_abstain) later.
		if self.use_cuda:
			p_out_abstain = torch.min(p_out_abstain,
				Variable(torch.Tensor([1. - epsilon])).cuda(self.cuda_device))
		else:
			p_out_abstain = torch.min(p_out_abstain,
				Variable(torch.Tensor([1. - epsilon])))

		return h_c, p_out, p_out_abstain


	def get_dac_loss(self, h_c, p_out_abstain):
		return (1. - p_out_abstain)*h_c - self.alpha_var*torch.log(1. - p_out_abstain)


	def print_abst_stats(self, epoch):
		print("\n##### Epoch %d abst_rate_ewma %f" 
			%(epoch, self.abst_rate_ewma))


	def print_inst_control_stats(self, epoch, control):
		# print("\nEpoch %d control %f abst_rate %f alpha_thresh %f alpha_var %f pid components %s" 
		# 	%(epoch, control, self.abst_rate, self.alpha_thresh.data, self.alpha_var, self.pid.components))
		print("\nEpoch %d control %f abst_rate %f alpha_thresh %f alpha_var %f pid components %s" 
			%(epoch, control, self.abst_rate, self.alpha_thresh.data, self.alpha_var, self.pid.components()))


	def update_alpha(self, control, epoch):
		#self.alpha_var = control

		#if error is +ve, abstention is not high enough. decrease alpha.
		#pdb.set_trace()
		assert(self.alpha_var >= 0.)
		self.alpha_var -= control
		if self.alpha_var < 0.:
			self.alpha_var = 0.


		if self.alpha_check_epoch is None:
			self.alpha_check_epoch = epoch

		else:
			if epoch > self.alpha_check_epoch: 
				if self.final_abst_rate is not None:
					self.update_pid_abst_set_point(self.abst_delta)
				self.alpha_check_epoch = epoch


	def __call__(self, input_batch, target_batch, epoch):

		if not self.model.training: #test mode. just return cross-entropy loss
			return F.cross_entropy(input_batch, target_batch)

		else: #in training mode.
			h_c, p_out, p_out_abstain = self.get_loss_terms(input_batch,target_batch)			
			#abstention rate update
			self.update_abst_rate(p_out)
			#update instantaneous alpha_thresh
			self.update_alpha_thresh(h_c, p_out_abstain)
			
			if epoch <= self.learn_epochs: #warm-up phase, so use regular cross-entropy
				# print("\nloss details (pre abstention): %d,%f,%f,%f,%f\n" %(epoch,p_out_abstain.mean(),loss.mean(),h_c.mean(),
				# 	self.alpha_thresh_ewma))
				return F.cross_entropy(input_batch, target_batch)

			else:#we are in abstaining phase.
				# if alpha has not been initialized, set it based on threshold.
				if self.alpha_var is None: 
					self.alpha_var = self.alpha_thresh_ewma
					control = 0.
				else: #do PID control
				#get PID controller value and set alpha to this 
					control = self.pid(self.abst_rate_ewma)
					#control = self.pid(self.abst_rate)
					self.update_alpha(control, epoch)
		
				#TODO: comment out before pusing to github
				#self.print_inst_control_stats(epoch, control)
				return self.get_dac_loss(h_c, p_out_abstain).mean()
				# print("\nloss details (during abstention): %d, %f,%f,%f,%f\n" %(epoch,p_out_abstain.mean(), h_c.mean(),
	 		  #   		self.alpha_thresh_ewma, self.alpha_var))



# loss_fn_dict = {
# 	'dac_loss_pid' : dac_loss_pid
# 	}		
