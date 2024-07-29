import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
            #    'Implement initialization of variances, means, pi_k using k-means')
            k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            self.means, membership, runs = k_means.fit(x) 
            self.variances = np.zeros((self.n_cluster,D,D))
            self.pi_k = np.zeros(self.n_cluster)
              
            for k in range(self.n_cluster):
                x_idx = np.where(membership == k) # indices of x labelled as k
                #print('x_idx', x_idx)
                Nk = len(x_idx[0])
                self.pi_k[k] = (Nk * 1.0 / N)
                self.variances[k] = np.sum([np.outer(x[j]-self.means[k],x[j]-self.means[k]) for j in x_idx[0]],axis=0)
                self.variances[k] = self.variances[k] * 1.0 / Nk
                
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means = np.random.rand(self.n_cluster,D)
            self.variances = np.zeros((self.n_cluster,D,D))
            self.pi_k = np.zeros(self.n_cluster)
            for k in range(self.n_cluster):
                self.variances[k] = np.identity(D)
                self.pi_k[k] = 1.0 / self.n_cluster
           

            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement fit function (filename: gmm.py)')
                
        l = self.compute_log_likelihood(x,self.means,self.variances,self.pi_k)
        rik = np.zeros((N,self.n_cluster)) # N*K matrix
        for iteration in range(self.max_iter):        
            for k in range(self.n_cluster):
                variances_k = self.variances[k]
                means_k = self.means[k] 
                while np.linalg.matrix_rank(variances_k) < D:
                    variances_k += 0.001 * np.identity(D)
                variances_k_inv = np.linalg.inv(variances_k)
                variances_k_det = np.linalg.det(variances_k)
                means_k = np.matmul(np.ones((N,1)),means_k.reshape(1,D))
                diff = x - means_k
                i = -0.5*np.sum( np.matmul(diff, variances_k_inv) * diff, 1)
                rik[:,k] = self.pi_k[k]*np.exp(i) / (np.sqrt((2 * np.pi)**D * variances_k_det))
            rik = rik/np.sum(rik,1)[:,None] 
            #finished E step, start M step
            Nk = np.sum(rik,0)
            # equation 5
            self.pi_k = Nk/N # equation 8
            self.means = np.matmul(rik.T, x)/Nk[:,None] 
            for k in range(self.n_cluster):
                means_k = self.means[k]
                self.variances[k] = (1.0 / Nk[k]) * np.sum([rik[i][k] * np.outer(x[i]-means_k, x[i]-means_k) for i in range(N)], 0)
                #equation 7
            l_new = self.compute_log_likelihood(x,self.means,self.variances,self.pi_k)
            if (abs(l_new-l) <= self.e):
                break
            else:
                l = l_new
        return iteration+1
        """
        N,D = x.shape
        sum1 = np.zeros((N,self.n_cluster)) # N K
        for k in range(self.n_cluster):
            variances_k = variances[k] # K D D
            means_k = means[k] # K D
            while np.linalg.matrix_rank(variances_k) < D:
                variances_k += 0.001 * np.identity(D)
            variances_k_inv = np.linalg.inv(variances_k)
            variances_k_det = np.linalg.det(variances_k)
            means_k = np.matmul(np.ones((N,1)),means_k.reshape(1,D))
            diff = x - means_k
            i = -0.5 * np.sum( np.matmul(diff, variances_k_inv) * diff, 1)
            sum1[:,k] = pi_k[k] * np.exp(i) / (np.sqrt((2*np.pi)**D * variances_k_det))
            # sum over k
        log_likelihood = float(np.sum(np.log(np.sum(sum1,1))))
        """
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement sample function in gmm.py')
        """
        samples = np.zeros((N,self.means.shape[1]))
        for i in range(N):
            k = np.argmax(np.random.multinomial(1,self.pi_k,1))
            samples[i] = np.random.multivariate_normal(self.means[k],self.variances[k])
      
        """
        
        sample_k = np.random.choice(range(0, self.n_cluster), N, replace=True, p = self.pi_k)
        samples = np.zeros((N,self.means.shape[1])) # N*D matrix
        for i in range(N):
            k = int(sample_k[i])
            
            samples[i] = np.random.multivariate_normal(self.means[k],self.variances[k])
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement compute_log_likelihood function in gmm.py')
        N,D = x.shape
        sum1 = np.zeros((N,self.n_cluster)) # N K
        for k in range(self.n_cluster):
            variances_k = variances[k] # K D D
            means_k = means[k] # K D
            while np.linalg.matrix_rank(variances_k) < D:
                variances_k += 0.001 * np.identity(D)
            variances_k_inv = np.linalg.inv(variances_k)
            variances_k_det = np.linalg.det(variances_k)
            means_k = np.matmul(np.ones((N,1)),means_k.reshape(1,D))
            diff = x - means_k
            i = -0.5 * np.sum( np.matmul(diff, variances_k_inv) * diff, 1)
            sum1[:,k] = pi_k[k] * np.exp(i) / (np.sqrt((2*np.pi)**D * variances_k_det))
            # sum over k
        log_likelihood = float(np.sum(np.log(np.sum(sum1,1))))
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            D = self.variance.shape[0]
            while (np.linalg.det(self.variance)==0):
                self.variance += 0.001 * np.identity(D)
            self.inv = np.linalg.inv(self.variance)
            self.c = np.linalg.det(self.variance) * (2 * np.pi)**D
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            
            p = np.exp(-0.5 * np.dot(np.dot(x - self.mean, self.inv), x - self.mean)) 
            p = p / np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
