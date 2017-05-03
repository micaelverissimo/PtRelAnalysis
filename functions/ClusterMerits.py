import numpy as np
from numpy import linalg as LA
from scipy.stats.mstats import gmean

# use to calculate the baricenter of a class or all data set
def Baricenter(data):
    dataCenter = np.sum(data,axis=0)/float(data.shape[0])
    return dataCenter

# use to calculate the total dispersion of a class or all data set F_0 or the mean dispersion
def DispTotal(data,dataCenter,media = 'm'):
    dispTotal = 0
    for i in range(data.shape[0]):
        dispTotal += LA.norm(data[i,:]-dataCenter) ** 2
    if (media == 'm'):
        return dispTotal/data.shape[0]
    else:
        return dispTotal

# use to calculate the diameter of the class
def ClassDiameter(data,dataCenter):
    phi = []
    for i in range(data.shape[0]):
        phi.append(LA.norm(data[i,:]-dataCenter))
    return max(phi)

# use to calculate the intra class dispersion F_in
def DispIntraClass(list_of_cluster):
    dispIntra = 0
    for i in range(len(list_of_cluster)):
        dispIntra += DispTotal(list_of_cluster[i],Baricenter(list_of_cluster[i]),media='')
    return dispIntra

# use to calculate the inter class dispersion F_out
def DispInterClass(list_of_cluster,dataSet):
    DispInter = 0
    for i in range(len(list_of_cluster)):
        DispInter += list_of_cluster[i].shape[0]*(LA.norm(Baricenter(list_of_cluster[i])-Baricenter(dataSet))**2)
    return DispInter

# ================================================================== acc and SP ===================================
# measure accuracy of each class
def Acc(output,true_label):
    acc = np.zeros(len(np.unique(output)))
    for i in range(len(np.unique(output))):
        acc[i] = float(np.sum(output[true_label==i]==i))/(np.sum(true_label==i))
    return acc

# Calculate the sum-product
def SP(output,true_label):
    
    SP = np.power(np.mean(Acc(output,true_label))*gmean(Acc(output,true_label)),0.5)
    
    return SP