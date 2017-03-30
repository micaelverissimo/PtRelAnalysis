import numpy as np
from numpy import linalg as LA

# use to calculate the baricenter of a class or all data set
def Baricenter(data):
    dataCenter = np.sum(data,axis=0)/data.shape[0]
    print 'Baricenter of set: ', dataCenter
    return dataCenter

# use to calculate the total dispersion of a class or all data set
def DispTotal(data,dataCenter,media = 'm'):
    dispTotal = 0
    for i in range(data.shape[0]):
        dispTotal += LA.norm(data[i,:]-dataCenter) ** 2
    if (media == 'm'):
        print 'Mean Total Dispersion of set: ', dispTotal/data.shape[0]
        return dispTotal/data.shape[0]
    else:
        print 'Total Dispersion of set: ', dispTotal
        return dispTotal

# use to calculate the diameter of the class
def ClassDiameter(data,dataCenter):
    phi = []
    for i in range(data.shape[0]):
        phi.append(LA.norm(data[i,:]-dataCenter))
    print 'Class Diameter: ', max(phi)
    return max(phi)