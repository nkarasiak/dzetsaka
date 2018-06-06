'''!@brief Accuracy index by Mathieu Fauvel
'''

import numpy as np

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
        n = yp.size
        C=int(yr.max())
        self.confusion_matrix=np.zeros((C,C))
        
        # Compute confusion matrix
        for i in range(n):
            self.confusion_matrix[yp[i].astype(int)-1,yr[i].astype(int)-1] +=1
        
        # Compute overall accuracy
        self.OA=np.sum(np.diag(self.confusion_matrix))/n
        
        # Compute Kappa
        nl = np.sum(self.confusion_matrix,axis=1)
        nc = np.sum(self.confusion_matrix,axis=0)
        self.Kappa = ((n**2)*self.OA - np.sum(nc*nl))/(n**2-np.sum(nc*nl))
        
        # TBD Variance du Kappa
        
    
    
class statsFromConfusionMatrix:
    def __init__(self,confusionMatrix):
        self.confusionMatrix = confusionMatrix
        self.n = np.sum(self.confusionMatrix)
        self.OA = self.__get_OA()
        self.kappa = self.__get_kappa()
        self.F1mean = self.__get_F1Mean()
        self.F1 = self.__get_F1()
        
    def __get_OA(self):
        """
        Compute overall accuracy
        """
        return np.sum(np.diag(self.confusionMatrix))/float(self.n)
    def __get_kappa(self):
        """
        Compute Kappa
        """
        nl = np.sum(self.confusionMatrix,axis=1)
        nc = np.sum(self.confusionMatrix,axis=0)
        OA = np.sum(np.diag(self.confusionMatrix))/float(self.n)
        return ((self.n**2)*OA - np.sum(nc*nl))/(self.n**2-np.sum(nc*nl))
    def __get_F1Mean(self):
        """
        Compute F1 Mean
        """
        nl = np.sum(self.confusionMatrix,axis=1,dtype=float)
        nc = np.sum(self.confusionMatrix,axis=0,dtype=float)
        return 2*np.mean( np.divide( np.diag(self.confusionMatrix), (nl + nc)) )
    def __get_F1(self):
        """
        Compute F1 per class
        """
        f1 = []
        for label in range(self.confusionMatrix.shape[0]):
            TP = self.confusionMatrix[label,label]
            #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
            FN = np.sum(self.confusionMatrix[:,label])-self.confusionMatrix[label,label]
            FP = np.sum(self.confusionMatrix[label,:])-self.confusionMatrix[label,label]
        
            f1.append(2*TP / (2*TP+FP+FN))
        return f1