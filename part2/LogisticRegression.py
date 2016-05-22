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
		self.X = X
		self.mean_x = X.mean(axis=0)
		self.std_x = X.std(axis=0)
		self.features = np.ones((X.shape[0],X.shape[1]+1))
		self.features[:,1:] = (X-self.mean_x)/self.std_x
		self.shuffled_features = self.features
		self.shuffled_labels = self.labels
		self.likelihood_history = []



	def log_likelihood(self):
		"""Calculate the loglikelihood for the current set of weights and features.
 
		Returns
		-------
		out : float
		""" 
		#Get Probablities
		p = self.probability()
		#Get Log Likelihood For Each Row of Dataset
		loglikelihood = self.labels*np.log(p+1e-24) + (1-self.labels)*np.log(1-p+1e-24)
		#Return Sum
		return -1*loglikelihood.sum()

	def probability(self):
		"""Computes the logistic probability of being a positive example
		
		Returns
		-------
		out : ndarray (1,)
			Probablity of being a positive example
		"""
		return 1/(1+np.exp(-self.features.dot(self.w)))

	def log_likelihood_gradient(self):
		"""Calculate the loglikelihood gradient for the current set of weights and features.
 
		Returns
		-------
		out : ndarray(n features, 1)
			gradient of the loglikelihood
		""" 
		error = self.labels-self.probability()
		product = error*self.features
		return product.sum(axis=0).reshape(self.w.shape)

	def gradient_decent(self,alpha=1e-7,max_iterations=1e4):
		"""Runs the gradient decent algorithm
		
		Parameters
		----------
		alpha : float
			The learning rate for the algorithm

		max_iterations : int
			The maximum number of iterations allowed to run before the algorithm terminates
			
		"""
		previous_likelihood = self.log_likelihood()
		difference = self.tolerance+1
		iteration = 0
		self.likelihood_history = [previous_likelihood]
		while (difference > self.tolerance) and (iteration < max_iterations):
			self.w = self.w + alpha*self.log_likelihood_gradient()
			temp = self.log_likelihood()
			difference = np.abs(temp-previous_likelihood)
			previous_likelihood = temp
			self.likelihood_history.append(previous_likelihood)
			iteration += 1
	
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
		return 1/(1+np.exp(-self.shuffled_features[row,:].dot(self.w)))
	
	
	
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
		error = self.shuffled_labels[row]-self.row_probability(row)
		product = self.shuffled_features[row,:]*error
		return product.reshape(self.features.shape[1],1)
		
	def stocastic_gradient_decent(self,alpha=0.1,max_iterations=1e2):
		"""Runs the gradient decent algorithm
		
		Parameters
		----------
		alpha : float
			The learning rate for the algorithm

		max_iterations : int
			The maximum number of iterations allowed to run before the algorithm terminates
			
		"""
		difference = self.tolerance+1.
		previous_likelihood = self.log_likelihood()
		rows = range(len(self.features))
		np.random.shuffle(rows)
		self.shuffled_features = self.shuffled_features[rows,:]
		self.shuffled_labels = self.shuffled_labels[rows]
		iteration = 0
		self.likelihood_history = [previous_likelihood]
		
		while (difference > self.tolerance) & (iteration < max_iterations):
			for i in xrange(len(self.features)):
				self.w = self.w + alpha*self.row_log_likelihood_gradient(i)
			temp = self.log_likelihood()
			difference = np.abs(temp - previous_likelihood)
			
			#print previous_likelihood, temp, difference
			
			previous_likelihood = temp    
			
			np.random.shuffle(rows)
			self.shuffled_features = self.shuffled_features[rows,:]
			self.shuffled_labels = self.shuffled_labels[rows]
			iteration += 1
			self.likelihood_history.append(previous_likelihood)

	
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
		features = np.ones((X.shape[0],X.shape[1]+1))
		features[:,1:] = (X-self.mean_x)/self.std_x
		return 1/(1+np.exp(-features.dot(self.w)))

	def get_coefficients(self):
		new_coef = self.w.T[0]/np.hstack((1,self.std_x))
		new_coef[0] = self.w.T[0][0]-(self.mean_x*self.w.T[0][1:]/self.std_x).sum()
		return new_coef