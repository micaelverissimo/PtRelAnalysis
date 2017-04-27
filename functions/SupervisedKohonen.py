"""
    Implementation of Supervised Kohonen NN
"""

import numpy as np

class TrainParameters(object):
    
    def __init__(self,learning_rate=0.1,verbose= False):
        self.learning_rate = learning_rate
        self.verbose = verbose
        
    def Print(self):
        print 'Supervised Kohonen NN TrainParameters'
        print 'Learning Rate: %1.5f'%(self.learning_rate)
        if self.verbose:
            print 'Verbose: True'
        else:
            print 'Verbose: False'
            
class KohonenNN(object):
    
    """
    Supervised Kohonen NN class
    This class implement the Supervised Kohonen Neural Network
    """
    def __init__(self, n_sinapses=2, dist='euclidean', randomize=True, dev=True):
        """
        Supervised Kohonen NN constructor
            n_sinapses: Number of sinapses to be used (default: 2)
            similarity_radius: Similarity Radius (default: 0.1) ~ notNOW
            dist: distance method used (defaults: euclidean)
            randonize: do or not random access to data
            dev: Development flag
        """
        self.n_sinapses = n_sinapses
        self.sinapses = None
        self.dist = dist
        self.randomize = randomize
        self.dev = dev
        
    def calc_dist(self, pt1, pt2):
        if self.dist == 'euclidean':
            return np.linalg.norm((pt1-pt2), ord=2)
        
    # not working
    def update_sinapses(self, sinapse_id, event, trn_params):
        self.sinapses[sinapse_id,:] = (self.sinapses[sinapse_id,:]+
                                       trn_params.learning_rate*
                                       (event-self.sinapses[sinapse_id,:]))
        
    def fit(self, data, label, trn_params = None, sinapses=None):
        if trn_params is None:
            trn_params = TrainParameters()
        
        if self.dev:
            trn_params.Print()
            
        if self.randomize:
            if data.shape[0] < data.shape[1]:
                trn_data = data[:,np.random.permutation(data.shape[1])].T
            else:
                trn_data = data[np.random.permutation(data.shape[0]),:]
        else:
            if data.shape[0] < data.shape[1]:
                trn_data = data.T
            else:
                trn_data = data
        print "Number of events:",trn_data.shape[0]
        
        if self.sinapses is None:
            self.sinapses = np.array([[-3.342,  -2.433 ],[3.465,  2.456]])
            
        for ievent in range(trn_data.shape[0]):
            self.update_sinapses(label[ievent],trn_data[ievent,:],trn_params = trn_params)
            
        return self.sinapses