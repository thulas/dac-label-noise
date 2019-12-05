"""
loss function definitions for deep abstaining classifier.
"""



import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import pdb
import math

#for numerical stability
epsilon = 1e-7



#loss calculation and alpha-ramping are rolled into one function. This is invoked
#after every iteration
class dac_loss(_Loss):
	def __init__(self, model, learn_epochs, total_epochs, use_cuda=False, cuda_device=None, 
			alpha_final=1.0,alpha_init_factor=64.):
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


	def __call__(self, input_batch, target_batch, epoch):
		if epoch <= self.learn_epochs or not self.model.training:
			#pdb.set_trace()
			loss =  F.cross_entropy(input_batch, target_batch, reduction='none')
			#return loss.mean()
			if self.model.training:
				h_c = F.cross_entropy(input_batch[:,0:-1],target_batch,reduction='none')
				p_out = torch.exp(F.log_softmax(input_batch,dim=1))
				p_out_abstain = p_out[:,-1]
				#pdb.set_trace()

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
			#calculate cross entropy only over true classes
			h_c = F.cross_entropy(input_batch[:,0:-1],target_batch,reduce=False)
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

			#update instantaneous alpha_thresh
			self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)

			#if (epoch == 5):
			#	pdb.set_trace()
			try:
	    		#update alpha_thresh_ewma
				if self.alpha_thresh_ewma is None:
					self.alpha_thresh_ewma = self.alpha_thresh #Variable(((1. - p_out_abstain)*h_c).mean().data)
				else:
					self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
						(1. - self.ewma_mu)*self.alpha_thresh_ewma.data)


				if self.alpha_var is None: #hasn't been initialized. do it now
					#we create a freshVariable here so that the history of alpha_var
					#computation (which depends on alpha_thresh_ewma) is forgotten. This
					#makes self.alpha_var a leaf variable, which will not be differentiated.
					#aggressive initialization of alpha to jump start abstention
					self.alpha_var = 	Variable(self.alpha_thresh_ewma.data /self.alpha_init_factor)
					self.alpha_inc =  (self.alpha_final - self.alpha_var.data)/(self.total_epochs - epoch)
					self.alpha_set_epoch = epoch

				else:		
					# we only update alpha every epoch
					if epoch > self.alpha_set_epoch: 
						self.alpha_var = Variable(self.alpha_var.data + self.alpha_inc)
						self.alpha_set_epoch = epoch

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


# loss_fn_dict = {
# 	'dac_loss' : dac_loss
#	}		
