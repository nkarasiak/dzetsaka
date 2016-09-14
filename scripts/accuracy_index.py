'''!@brief Accuracy index by Mathieu Fauvel
'''
import scipy as sp

class CONFUSION_MATRIX:
    def __init__(self):
        self.confusion_matrix=None
        self.OA=None
        self.Kappa = None
        
    def compute_confusion_matrix(self,yp,yr):
        ''' 
        Compute the confusion matrix
        '''
        # Initialization
        n = yp.shape[0]
        C=int(yr.max())
        self.confusion_matrix=sp.zeros((C+1,C+1))
        self.confusion_matrix[0]=range(0,int(self.confusion_matrix[0].shape[0]))
        self.confusion_matrix[:,0]=range(0,int(self.confusion_matrix[0].shape[0]))
        
        
        # Compute confusion matrix
        for i in range(n):
            self.confusion_matrix[yp[i].astype(int),yr[i].astype(int)] +=1
        
        # Compute overall accuracy
        self.OA=sp.sum(sp.diag(self.confusion_matrix))/n
        
        # Compute Kappa
        nl = sp.sum(self.confusion_matrix,axis=1)
        nc = sp.sum(self.confusion_matrix,axis=0)
        self.Kappa = ((n**2)*self.OA - sp.sum(nc*nl))/(n**2-sp.sum(nc*nl))
        
        # TBD Variance du Kappa