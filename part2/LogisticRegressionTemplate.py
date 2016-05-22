import numpy as np
class LogisticRegression:
	
	def __init__(self,X,y,tolerance=1e-5):
		"""Initializes Class for Logistic Regression
		
		Parameters
		----------
		X : ndarray(n-rows,m-features)
			Numerical training data.

		y: ndarray(n-rows,)
			Interger training labels.
			
		tolerance : float (default 1e-5)
			Stopping threshold difference in the loglikelihood between iterations.
			
		"""
		self.tolerance = tolerance
		self.labels = y.reshape(y.size,1)
		#create weights equal to zero with an intercept coefficent at index 0
		self.w = np.zeros((X.shape[1]+1,1))
		#Add Intercept Data Point of 1 to each row
		self.features = np.ones((X.shape[0],X.shape[1]+1))
		self.features[:,1:] = X
		self.shuffled_features = self.features
		self.shuffled_labels = self.labels
		self.likelihood_history = []

	def probability(self):
		"""Computes the logistic probability of being a positive exampe
		
		Returns
		-------
		out : ndarray (1,)
			Probablity of being a positive example
		"""
		pass

	def log_likelihood(self):
		"""Calculate the loglikelihood for the current set of weights and features.
 
		Returns
		-------
		out : float
		""" 
		pass



	def log_likelihood_gradient(self):
		"""Calculate the loglikelihood gradient for the current set of weights and features.
 
		Returns
		-------
		out : ndarray(n features, 1)
			gradient of the loglikelihood
		""" 
		pass

	def gradient_decent(self,alpha=1e-7,max_iterations=1e4):
		"""Runs the gradient decent algorithm
		
		Parameters
		----------
		alpha : float
			The learning rate for the algorithm

		max_iterations : int
			The maximum number of iterations allowed to run before the algorithm terminates
			
		"""
		pass
	
	def row_probability(self,row):
		"""Computes the logistic probability of being a positive example for a given row
		
		Parameters
		----------
		row : int
			Row from feature matrix with to calculate the probablity.
			
		Returns
		-------
		out : ndarray (1,)
			Probablity of being a positive example
		"""
		pass
	
	
	
	def row_log_likelihood_gradient(self,row):
		"""Computes the loglikelihood gradient for a given row
		
		Parameters
		----------
		row : int
			Row from feature matrix with to calculate the probablity.
			
		Returns
		-------
		out : ndarray(n features, 1)
			gradient of the loglikelihood
		"""
		pass
		
	def stocastic_gradient_decent(self,alpha=0.1,max_iterations=1e2):
		"""Runs the gradient decent algorithm
		
		Parameters
		----------
		alpha : float
			The learning rate for the algorithm

		max_iterations : int
			The maximum number of iterations allowed to run before the algorithm terminates
			
		"""
		pass
	
	def predict_probabilty(self,X):
		"""Computes the logistic probability of being a positive example
		
		Parameters
		----------
		X : ndarray (n-rows,n-features)
			Test data to score using the current weights

		Returns
		-------
		out : ndarray (1,)
			Probablity of being a positive example
		"""
		pass