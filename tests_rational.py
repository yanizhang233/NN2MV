from NN2MV import NN2MVRational
import numpy as np

## test 1, sigma(0.5x-0.5y), with precision s=2
Phi1 = []
Phi1.append(  np.array([[.5, -0.5],[.5, -0.5]]) )
Phi1.append(  np.array([[0.0], [-1]]) )
Phi1.append(  np.array([[-1, 1]]) )
Phi1.append(  np.array([[1]]) )

nn2mv = NN2MVRational(Phi1, 2)
nn2mv.extract()


## test 2
Phi2 = []
Phi2.append(  np.array([[.5, -0.5, 0.5],[.5, -0.5, -0.5]]) )
Phi2.append(  np.array([[0.0], [-1]]) )
Phi2.append(  np.array([[-1, 1]]) )
Phi2.append(  np.array([[1]]) )
nn2mv = NN2MVRational(Phi2, 3)
nn2mv.extract()