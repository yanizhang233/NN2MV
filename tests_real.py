from NN2MV import NN2MVReal
import numpy as np

## test 1
Phi1 = []
Phi1.append(  np.array([[.5, -0.5],[.5, -0.5]]) )
Phi1.append(  np.array([[0.0], [-1]]) )
Phi1.append(  np.array([[.5, -0.5],[.5, -0.5],[.3, -0.26]]) )
Phi1.append(  np.array([[0.0], [-1], [-2.8]]) )
Phi1.append(  np.array([[1, -1, -1]]) )
Phi1.append(  np.array([[.0]]) )

nn2mv = NN2MVReal(Phi1)
nn2mv.extract()

## test 2
Phi2 = []
Phi2.append(  np.array([[.5, -0.5],[.5, -0.5]]) )
Phi2.append(  np.array([[0.0], [-1]]) )
Phi2.append(  np.array([[1, -1]]) )
Phi2.append(  np.array([[.0]]) )
nn2mv = NN2MVReal(Phi2)
nn2mv.extract()


## test 3
Phi3 = []
Phi3.append(  np.array([[.5, -0.5],[.5, -0.5]]) )
Phi3.append(  np.array([[0.0], [-1]]) )
Phi3.append(  np.array([[1, -1]]) )
Phi3.append(  np.array([[.0]]) )
nn2mv = NN2MVReal(Phi3)
nn2mv.extract()
