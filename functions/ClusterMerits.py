import numpy as np
from numpy import linalg as LA

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