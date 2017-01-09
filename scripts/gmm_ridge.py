"""!@brief Script by Mathieu Fauvel which performs Gaussian Mixture Model
"""# -*- coding: utf-8 -*-
import scipy as sp
from scipy import linalg
import multiprocessing as mp

## Temporary predict function
def predict(tau,model,xT,yT):
    err = sp.zeros(tau.size)
    for j,t in enumerate(tau):
        yp = model.predict(xT,tau=t)[0]
        eq = sp.where(yp.ravel()==yT.ravel())[0]
        err[j] = eq.size*100.0/yT.size
    return err
    

class CV:
    '''
    This class implements the generation of several folds to be used in the cross validation
    '''
    def __init__(self):
        self.it=[]
        self.iT=[]

    def split_data(self,n,v=5):
        ''' The function split the data into v folds. Whatever the number of sample per class
        Input:
            n : the number of samples
            v : the number of folds
        Output: None        
        '''
        step = n //v  # Compute the number of samples in each fold
        sp.random.seed(1)   # Set the random generator to the same initial state
        t = sp.random.permutation(n)    # Generate random sampling of the indices
        
        indices=[]
        for i in range(v-1):            # group in v fold
            indices.append(t[i*step:(i+1)*step])
        indices.append(t[(v-1)*step:n])
                
        for i in range(v):
            self.iT.append(sp.asarray(indices[i]))
            l = range(v)
            l.remove(i)
            temp = sp.empty(0,dtype=sp.int64)
            for j in l:            
                temp = sp.concatenate((temp,sp.asarray(indices[j])))
            self.it.append(temp)

    def split_data_class(self,y,v=5):
        ''' The function split the data into v folds. The samples of each class are split approximatly in v folds
        Input:
            n : the number of samples
            v : the number of folds
        Output: None
        '''
        # Get parameters
        n = y.size
        C = y.max().astype('int')
       
        # Get the step for each class
        tc = []
        for j in range(v):
            tempit = []
            tempiT = []
            for i in range(C):
                # Get all samples for each class
                t  = sp.where(y==(i+1))[0]
                nc = t.size
                stepc = nc // v # Step size for each class
                if stepc == 0:
                    print "Not enough sample to build "+ str(v) +" folds in class " + str(i)                                    
                sp.random.seed(i)   # Set the random generator to the same initial state
                tc = t[sp.random.permutation(nc)] # Random sampling of indices of samples for class i
                        
                # Set testing and training samples
                if j < (v-1):
                    start,end = j*stepc,(j+1)*stepc
                else:
                    start,end = j*stepc,nc
                tempiT.extend(sp.asarray(tc[start:end])) #Testing
                k = range(v)
                k.remove(j)
                for l in k:
                    if l < (v-1):
                        start,end = l*stepc,(l+1)*stepc
                    else:
                        start,end = l*stepc,nc
                    tempit.extend(sp.asarray(tc[start:end])) #Training

            self.it.append(tempit)
            self.iT.append(tempiT)
            


class GMMR:
    def __init__(self):
        self.ni = []
        self.prop = []
        self.mean = []
        self.cov =[]
        self.Q = []
        self.L = []
        self.classnum = [] # to keep right labels
        self.tau = 0.0

    def learn(self,x,y):
        '''
        Function that learns the GMM with ridge regularizationb from training samples
        Input:
            x : the training samples
            y :  the labels
        Output:
            the mean, covariance and proportion of each class, as well as the spectral decomposition of the covariance matrix
        '''

        ## Get information from the data
        C = sp.unique(y).shape[0]
        #C = int(y.max(0))  # Number of classes
        n = x.shape[0]  # Number of samples
        d = x.shape[1]  # Number of variables
        eps = sp.finfo(sp.float64).eps

        ## Initialization
        self.ni = sp.empty((C,1))    # Vector of number of samples for each class
        self.prop = sp.empty((C,1))  # Vector of proportion
        self.mean = sp.empty((C,d))  # Vector of means
        self.cov = sp.empty((C,d,d)) # Matrix of covariance
        self.Q = sp.empty((C,d,d)) # Matrix of eigenvectors
        self.L = sp.empty((C,d)) # Vector of eigenvalues
        self.classnum = sp.empty(C).astype('uint8')
        
        ## Learn the parameter of the model for each class
        for c,cR in enumerate(sp.unique(y)):
            
            j = sp.where(y==(cR))[0]
            
            self.classnum[c] = cR # Save the right label
            self.ni[c] = float(j.size)    
            self.prop[c] = self.ni[c]/n
            self.mean[c,:] = sp.mean(x[j,:],axis=0)
            self.cov[c,:,:] = sp.cov(x[j,:],bias=1,rowvar=0)  # Normalize by ni to be consistent with the update formulae

            # Spectral decomposition
            L,Q = linalg.eigh(self.cov[c,:,:])
            idx = L.argsort()[::-1]
            self.L[c,:] = L[idx]
            self.Q[c,:,:]=Q[:,idx]

    def predict(self,xt,tau=None,confidenceMap=None):
        '''
        Function that predict the label for sample xt using the learned model
        Inputs:
            xt: the samples to be classified
        Outputs:
            y: the class
            K: the decision value for each class
        '''
        
        MAX = sp.finfo(sp.float64).max
        E_MAX = sp.log(MAX) # Maximum value that is possible to compute with sp.exp

        ## Get information from the data
        nt = xt.shape[0]        # Number of testing samples
        C = self.ni.shape[0]    # Number of classes

        ## Initialization
        K = sp.empty((nt,C))
        
        if tau is None:
            TAU=self.tau
        else:
            TAU=tau

        for c in range(C):
            invCov,logdet = self.compute_inverse_logdet(c,TAU)
            cst = logdet - 2*sp.log(self.prop[c]) # Pre compute the constant

            xtc = xt-self.mean[c,:]
            temp = sp.dot(invCov,xtc.T).T
            K[:,c] = sp.sum(xtc*temp,axis=1)+cst
            del temp,xtc        
        
        yp = sp.argmin(K,1)
            
        if confidenceMap is None:

            ## Assign the label save in classnum to the minimum value of K 
            yp = self.classnum[yp]
            
            return yp
            
        else:

            K *= -0.5
            K[K>E_MAX],K[K<-E_MAX] = E_MAX,-E_MAX
            sp.exp(K,out=K)
            K /= K.sum(axis=1).reshape(nt,1)
            K = K[sp.arange(len(K)),yp]
            #K = sp.diag(K[:,yp])
            
            yp = self.classnum[yp]
            
            return yp,K

    def compute_inverse_logdet(self,c,tau):
        Lr = self.L[c,:]+tau # Regularized eigenvalues
        temp = self.Q[c,:,:]*(1/Lr)
        invCov = sp.dot(temp,self.Q[c,:,:].T) # Pre compute the inverse
        logdet = sp.sum(sp.log(Lr)) # Compute the log determinant
        return invCov,logdet
    
    def BIC(self,x,y,tau=None):
        '''
        Computes the Bayesian Information Criterion of the model
        '''
        ## Get information from the data
        C,d = self.mean.shape
        n = x.shape[0]

        ## Initialization
        if tau is None:
            TAU=self.tau
        else:
            TAU=tau
            
        ## Penalization
        P = C*(d*(d+3)/2) + (C-1)
        P *= sp.log(n)
        
        ## Compute the log-likelihood
        L = 0
        for c in range(C):
            j = sp.where(y==(c+1))[0]
            xi = x[j,:]
            invCov,logdet = self.compute_inverse_logdet(c,TAU)
            cst = logdet - 2*sp.log(self.prop[c]) # Pre compute the constant
            xi -= self.mean[c,:]
            temp = sp.dot(invCov,xi.T).T
            K = sp.sum(xi*temp,axis=1)+cst
            L +=sp.sum(K)
            del K,xi

        return L + P

    def cross_validation(self,x,y,tau,v=5):
        ''' 
        Function that computes the cross validation accuracy for the value tau of the regularization
        Input:
            x : the training samples
            y : the labels
            tau : a range of values to be tested
            v : the number of fold
        Output:
            err : the estimated error with cross validation for all tau's value
        '''
        ## Initialization
        ns = x.shape[0]     # Number of samples
        np = tau.size       # Number of parameters to test
        cv = CV()           # Initialization of the indices for the cross validation
        cv.split_data_class(y)
        err = sp.zeros(np)  # Initialization of the errors

        ## Create GMM model for each fold
        model_cv = []
        for i in range(v):
            model_cv.append(GMMR())
            model_cv[i].learn(x[cv.it[i],:], y[cv.it[i]])

        ## Initialization of the pool of processes
        pool = mp.Pool()
        processes = [pool.apply_async(predict,args=(tau,model_cv[i],x[cv.iT[i],:],y[cv.iT[i]])) for i in range(v)]
        pool.close()
        pool.join()
        for p in processes:
            err += p.get()
        err /= v

        ## Free memory        
        for model in model_cv:
            del model
        elf
        del processes,pool,model_cv

        return tau[err.argmax()],err
