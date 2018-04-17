'''!@brief Accuracy index by Mathieu Fauvel
'''
from builtins import range
from builtins import object
import scipy as sp

class CONFUSION_MATRIX(object):
    def __init__(self):
        self.confusion_matrix=None
        self.OA=None
        self.Kappa = None
        
    def compute_confusion_matrix(self,yp,yr):
        ''' 
        Compute the confusion matrix
        '''
        # Initialization
        n = yp.size
        C=int(yr.max())
        self.confusion_matrix=sp.zeros((C,C))
        
        # Compute confusion matrix
        for i in range(n):
            self.confusion_matrix[yp[i].astype(int)-1,yr[i].astype(int)-1] +=1
        
        # Compute overall accuracy
        self.OA=sp.sum(sp.diag(self.confusion_matrix))/n
        
        # Compute Kappa
        nl = sp.sum(self.confusion_matrix,axis=1)
        nc = sp.sum(self.confusion_matrix,axis=0)
        self.Kappa = ((n**2)*self.OA - sp.sum(nc*nl))/(n**2-sp.sum(nc*nl))
        
        # TBD Variance du Kappa
        
    
    
class statsFromConfusionMatrix(object):
    def __init__(self,confusionMatrix):
        self.confusionMatrix = confusionMatrix
        self.n = sp.sum(self.confusionMatrix)
        self.OA = self.__get_OA()
        self.kappa = self.__get_kappa()
        self.F1 = self.__get_F1Mean()
        
    def __get_OA(self):
        """
        Compute overall accuracy
        """
        return sp.sum(sp.diag(self.confusionMatrix))/float(self.n)
    def __get_kappa(self):
        """
        Compute Kappa
        """
        nl = sp.sum(self.confusionMatrix,axis=1)
        nc = sp.sum(self.confusionMatrix,axis=0)
        OA = sp.sum(sp.diag(self.confusionMatrix))/float(self.n)
        return ((self.n**2)*OA - sp.sum(nc*nl))/(self.n**2-sp.sum(nc*nl))
    def __get_F1Mean(self):
        """
        Compute F1 Mean
        """
        nl = sp.sum(self.confusionMatrix,axis=1,dtype=float)
        nc = sp.sum(self.confusionMatrix,axis=0,dtype=float)
        return 2*sp.mean( sp.divide( sp.diag(self.confusionMatrix), (nl + nc)) )