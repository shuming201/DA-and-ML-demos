import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		
		features = np.array(features)
		N = features.shape[0]
		Hx = np.zeros((N))
		for i in range(self.T):
			Hx = Hx + np.array(self.clfs_picked[i].predict(features)) * self.betas[i]
		Hx = np.sign(Hx)
		return Hx.tolist()
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		
		N = len(features)
		D = np.ones(N) * (1 / N)
		for j in range(self.T):
			errors = []
			for k in range(len(list(self.clfs))):
				errors.append(np.dot(D, np.not_equal(labels, list(self.clfs)[k].predict(features))))
			et = min(errors)
			ht = errors.index(et)
			y = [int(x) for x in np.not_equal(labels, list(self.clfs)[ht].predict(features))]
			pred = [x if x == 1 else -1 for x in y]
			self.clfs_picked.append(list(self.clfs)[ht])
			beta = 0.5 * np.log((1 - et) / et)
			self.betas.append(beta)
			#print (pred)

			D = np.multiply(D, np.exp([x * beta for x in pred]))
			D = D / np.sum(D)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	