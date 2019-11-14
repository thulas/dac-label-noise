"""
loss function definitions for deep abstaining classifier.
"""



import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import pdb
import math
from simple_pid import PID

#for numerical stability
epsilon = 1e-7

#TODO: remove below this might be changed from inside the main script
# total_epochs = 200
# alpha_final = 1.0
# alpha_init_factor = 64.




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


#loss calculation and alpha-auto-tune are rolled into one function. This is invoked
#after every iteration
class dac_loss(_Loss):
	def __init__(self, model, learn_epochs, total_epochs, use_cuda=False, cuda_device=None, 
		abst_rate=None, final_abst_rate=None,alpha_final=1.0,alpha_init_factor=64.):
		print("using dac loss function\n")
		super(dac_loss, self).__init__()
		self.model = model
		#self.alpha = alpha
		self.learn_epochs = learn_epochs
		self.total_epochs = total_epochs
		self.alpha_final = alpha_final
		self.alpha_init_factor = alpha_init_factor
		self.use_cuda = use_cuda
		self.cuda_device = cuda_device
		#self.kappa = kappa #not used

		# if self.use_cuda:
		# 	self.alpha_var =  Variable(torch.Tensor([self.alpha])).cuda(self.cuda_device)
		# else:
		# 	self.alpha_var =  Variable(torch.Tensor([self.alpha]))

		self.alpha_var = None

		self.alpha_thresh_ewma = None   #exponentially weighted moving average for alpha_thresh
		self.alpha_thresh = None #instantaneous alpha_thresh
		self.ewma_mu = 0.05 #mu parameter for EWMA; 
		self.curr_alpha_factor  = None #for alpha initiliazation
		self.alpha_inc = None #linear increase factor of alpha during abstention phase
		self.alpha_set_epoch = None

		self.abst_rate = None #instantaneous abstention rate
		self.abst_rate_ewma = None # exponentially weighted moving averge of abstention

		self.pid = None
		self.final_abst_rate = None
		#PD controller for pre-specified abstention rate
		if abst_rate is not None:
			#pdb.set_trace()
			if abst_rate < 0.:
				raise ValueError("Invalid abstention rate of %f. Must be non-negative" %(new_abst_rate))
			#self.pid = PID(1.,0.5, 0., sample_time=None,setpoint=abst_rate)
			#self.pid.output_limits = (-0.1,0.1)
			self.pid = PID(-1.,-0.5, 0., sample_time=None,setpoint=abst_rate)
			self.pid.output_limits = (0.,None)

			if final_abst_rate is not None:
				#if total_epochs is None:
				#	raise ValueError("total epochs must be specified if final abstention rate is specied")
				#else:
				self.abst_delta = (abst_rate - final_abst_rate)/(self.total_epochs-self.learn_epochs)
				self.final_abst_rate = final_abst_rate

	def update_abst_rate(self, abst_delta):
		#pdb.set_trace()
		if self.pid is not None:
			new_abst_rate  = max(0.,self.pid.setpoint - abst_delta)
			# if new_abst_rate < 0.:
			# 	raise ValueError("Invalid abstention rate of %f. Must be non-negaitve" (%new_abst_rate))
			# else:
			self.pid.setpoint = new_abst_rate
			print("DAC updated abstention rate to %f" %(new_abst_rate))
		else:
			print("Warning: Cannot update abstention rate as PID has not been initialized")


	def __call__(self, input_batch, target_batch, epoch):
		#pdb.set_trace()
		#TODO: remove global
		#global total_epochs, alpha_final 
		#pdb.set_trace()
		if epoch < self.learn_epochs or not self.model.training:
			loss =  F.cross_entropy(input_batch, target_batch, reduce=False)
			#return loss.mean()
			if self.model.training:
				h_c = F.cross_entropy(input_batch[:,0:-1],target_batch,reduce=False)
				
				
				#p_out = F.softmax(F.log_softmax(input_batch,dim=1),dim=1)
				p_out = torch.exp(F.log_softmax(input_batch,dim=1))
				p_out_abstain = p_out[:,-1]
				#pdb.set_trace()

				#abstention rate update
				self.abst_rate = get_abst_rate(p_out)
				if self.abst_rate_ewma is None:
					self.abst_rate_ewma = self.abst_rate
				else:
					self.abst_rate_ewma = self.ewma_mu*self.abst_rate + (1-self.ewma_mu)*self.abst_rate_ewma

				#update instantaneous alpha_thresh
				self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)
				#update alpha_thresh_ewma 
				if self.alpha_thresh_ewma is None:
					self.alpha_thresh_ewma = self.alpha_thresh #Variable(((1. - p_out_abstain)*h_c).mean().data)
				else:
					# self.alpha_thresh_ewma = Variable(self.ewma_mu*((1. - p_out_abstain)*h_c).mean().data + \
					# 	(1. - self.ewma_mu)*self.alpha_thresh_ewma.data)
					self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
						(1. - self.ewma_mu)*self.alpha_thresh_ewma.data)

				# print("\nloss details (pre abstention): %d,%f,%f,%f,%f\n" %(epoch,p_out_abstain.mean(),loss.mean(),h_c.mean(),
				# 	self.alpha_thresh_ewma))
				
			return loss.mean()

		else:
			#pdb.set_trace()
			#calculate cross entropy only over true classes
			h_c = F.cross_entropy(input_batch[:,0:-1],target_batch,reduce=False)
			#p_out = F.softmax(F.log_softmax(input_batch,dim=1),dim=1)
			p_out = torch.exp(F.log_softmax(input_batch,dim=1))
			#probabilities of abstention  class
			p_out_abstain = p_out[:,-1]

			# avoid numerical instability by upper-bounding 
			# p_out_abstain to never be more than  1 - eps since we have to
			# take log(1 - p_out_abstain) later.
			# pdb.set_trace()
			if self.use_cuda:
				p_out_abstain = torch.min(p_out_abstain,
					Variable(torch.Tensor([1. - epsilon])).cuda(self.cuda_device))
			else:
				p_out_abstain = torch.min(p_out_abstain,
					Variable(torch.Tensor([1. - epsilon])))

	    	#loss = (1. - p_out_abstain)*h_c + 
	    	#	torch.log(1.+self.alpha_var)*p_out_abstain

			#abstention rate update
			self.abst_rate = get_abst_rate(p_out)
			if self.abst_rate_ewma is None:
				self.abst_rate_ewma = self.abst_rate
			else:
				self.abst_rate_ewma = self.ewma_mu*self.abst_rate + (1-self.ewma_mu)*self.abst_rate_ewma




			#update instantaneous alpha_thresh
			self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)

			#if (epoch == 5):
			#	pdb.set_trace()

			try:
	    		#update alpha_thresh_ewma
				if self.alpha_thresh_ewma is None:
					self.alpha_thresh_ewma = self.alpha_thresh #Variable(((1. - p_out_abstain)*h_c).mean().data)
				else:
					# self.alpha_thresh_ewma = Variable(self.ewma_mu*((1. - p_out_abstain)*h_c).mean().data + \
					# 	(1. - self.ewma_mu)*self.alpha_thresh_ewma.data)
					self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
						(1. - self.ewma_mu)*self.alpha_thresh_ewma.data)


				if self.alpha_var is None: #hasn't been initialized. do it now
			
					#we create a freshVariable here so that the history of alpha_var
					#computation (which depends on alpha_thresh_ewma) is forgotten. This
					#makes self.alpha_var a leaf variable, which will not be differentiated.
					

					#pdb.set_trace()
					#aggressive initialization of alpha to jump start abstention
					self.alpha_var = 	Variable(self.alpha_thresh_ewma.data /self.alpha_init_factor)
					self.alpha_inc =  (self.alpha_final - self.alpha_var.data)/(self.total_epochs - epoch)
					self.alpha_set_epoch = epoch

				else:		
					# we only update alpha every epoch
					#pass
					#self.alpha_var = Variable(self.alpha_thresh_ewma.data)

					if self.pid is not None:
						#delta = self.pid(self.abst_rate_ewma)
						control = self.pid(self.abst_rate_ewma)
						#print("control %f abst_rate %f abst_rate_ewma %f" %(control, self.abst_rate, self.abst_rate_ewma) )
						#pdb.set_trace()

						#self.alpha_var = Variable(self.alpha_thresh.data - .05)
						#self.alpha_var = Variable(torch.max(self.alpha_thresh_ewma.data - delta,torch.tensor(0.001).cuda()))
						try:
							self.alpha_var = Variable(torch.tensor(control).clone().detach())
						except TypeError:
							pdb.set_trace()

					else:
						control = 0.

					if epoch > self.alpha_set_epoch: 
						if self.pid is None:
							self.alpha_var = Variable(self.alpha_var.data + self.alpha_inc)
					# 	#self.alpha_var = Variable(self.alpha_var.data/2.)		
					# 	#self.alpha_var = Variable(self.alpha_thresh_ewma.data/0.8)				
						self.alpha_set_epoch = epoch
						if self.final_abst_rate is not None:
							self.update_abst_rate(self.abst_delta)
						# print("delta %f, abst_rate %f abst_rate_ewma %f alpha_thresh %f alpha_thresh_ewma %f alpha_var %f" 
						# 	%(delta, self.abst_rate, self.abst_rate_ewma,
						#  self.alpha_thresh.data, self.alpha_thresh_ewma.data,self.alpha_var))
						print("\ncontrol %f, abst_rate %f abst_rate_ewma %f alpha_thresh %f alpha_thresh_ewma %f alpha_var %f" 
						 	%(control, self.abst_rate, self.abst_rate_ewma,
						  self.alpha_thresh.data, self.alpha_thresh_ewma.data,self.alpha_var))

				loss = (1. - p_out_abstain)*h_c - \
		    		self.alpha_var*torch.log(1. - p_out_abstain)

		    	#calculate entropy of the posterior over the true classes.
				#h_p_true = 	-(F.softmax(input_batch[:,0:-1],dim=1) \
			    #	*F.log_softmax(input_batch[:,0:-1],dim=1)).sum(1)

				#loss = loss - self.kappa*h_p_true

				# print("\nloss details (during abstention): %d, %f,%f,%f,%f\n" %(epoch,p_out_abstain.mean(), h_c.mean(),
		  #   		self.alpha_thresh_ewma, self.alpha_var))

				return loss.mean()
	    
			except RuntimeError as e:
	    		#pdb.set_trace()
				print(e)



loss_fn_dict = {
	'dac_loss' : dac_loss
	}		
