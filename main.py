#########################################################################################
#  Implementation of a competitive swarm optimizer (CSO) for feature selction in python
#  See the details of CSO in the following paper
#  Cite as:S. Gu, R. Cheng, Y. Jin, Feature Selection for High-Dimensional Classification using a Competitive Swarm Optimizer, Soft Computing. 22 (2018) 811-822.
#  The source code CSO is implemented by Yang Xuesen 
#  If you have any questions about the code, please contact: 
#  Python Implementation:
#  Varenyam at varenyam17@gmail.com
#
#  Matlab Implementation: 
#  Yang Xuesen at 1348825332@qq.com
#  Institution: Shenzhen University
#########################################################################################

from typing import List

from scipy.io import loadmat
import numpy as np
import numpy.matlib as matlib

from fit_temp import func

class CSOFS: 
    def __init__(self):
        sonar = loadmat("sonar.mat")
        self.X, self.data, self.gnd = sonar['X'], sonar['data'], sonar['gnd']
        # n: dimensionality    
        self.n = np.size(self.data, 1) - 1
        # maxfe: maximal number of fitness evalutations
        self.maxfe = 100
        # runnum: the number of trail runs
        self.runnum = 30
        # m: population size
        self.m = 50
        self.lu = np.vstack((-5 * np.ones((1, self.n)), 5 * np.ones((1, self.n))))
        self.fitness = np.zeros((self.m, 1))
        self.phi = 0.1
        self.results = np.zeros((1, self.runnum))

    def run(self):
        for i in range(0, self.runnum):
            XRRmin = matlib.repmat(self.lu[0, :], self.m, 1)
            XRRmax = matlib.repmat(self.lu[1, :], self.m, 1)

            p = XRRmin + np.multiply((XRRmax - XRRmin), np.random.rand(self.m, self.n))
            bi_position = np.zeros((self.m, self.n))

            pop = self.sigmoid(p, [1, 0])

            randNum = np.random.rand(self.m, self.n)
            change_position = (pop > randNum)
            
            bi_position[change_position] = 1

            feature = []
            for i in range(0, self.m):
                feature = [idx for idx in range(len(bi_position[i])) if bi_position[i][idx] == 1]
                self.fitness[i] = func(self.data, feature)
                break
                
            break


    def sigmoid(self, x: np.ndarray, parms: List[int]):
        """
        This function computes fuzzy membership values using a sigmoidal membership function
        """

        f = 1 / (1 + np.exp(-parms[0] * (x - parms[1])))

        return f



def main():
    csofs = CSOFS()
    csofs.run()

if __name__ == "__main__":
    main()