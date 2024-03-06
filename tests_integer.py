from NN2MV import NN2MV
import numpy as np

## test 1, hat function
Phi1 = []
Phi1.append(  np.array([[2],[2], [2]]) )
Phi1.append(  np.array([[0], [-1], [-2]]) )
Phi1.append(  np.array([[1, -2, 1]]) )
Phi1.append(  np.array([[0]]) )
nn2mv = NN2MV(Phi1)
nn2mv.extract()

## test 2, linear pieces of g2
Phi2 = []
Phi2.append(  np.array([[-4],[-4]]) )
Phi2.append(  np.array([[2], [1]]) )
Phi2.append(  np.array([[1, -1]]) )
Phi2.append(  np.array([[0]]) )
nn2mv = NN2MV(Phi2)
nn2mv.extract()

## test 3, g2 our method deep, g \circ g
Phi3 = []
Phi3.append(  np.array([[2],[2], [2]]) )
Phi3.append(  np.array([[0], [-1],[-2]]) )
Phi3.append(  np.array([[1, -2, 1],  [-1, 2, -1]]) )
Phi3.append(  np.array([[0],[-1]]) )
Phi3.append(  np.array([[2, -2],[2, -2], [2, -2]]))
Phi3.append(  np.array([[0], [-1],[-2]]) )
Phi3.append(  np.array([[1, -2, 1]]) )
Phi3.append(  np.array([[0]]) )
nn2mv = NN2MV(Phi3)
nn2mv.extract()

## test 4, example in script ICML2024 section 2 (x or x)and (not y)
Phi4 = []
Phi4.append(  np.array([[-2, 0],[0, 1], [0, -1]]) )
Phi4.append(  np.array([[1], [0], [0]]) )
Phi4.append(  np.array([[-1, -1, 1]]) )
Phi4.append(  np.array([[1]]) )
Phi4.append(  np.array([[1]]) )
Phi4.append(  np.array([[0]]) )
nn2mv = NN2MV(Phi4)
nn2mv.extract()


# Phi2 = []
# Phi2.append(  np.array([[-2, 0],[0, -1], [0, 1]]) )
# Phi2.append(  np.array([[1], [1], [-1]]) )
# Phi2.append(  np.array([[-1, 1, -1]]) )
# Phi2.append(  np.array([[0]]) )
# Phi2.append(  np.array([[1]]) )
# Phi2.append(  np.array([[0]]) )
# nn2mv = NN2MV(Phi2)
# nn2mv.extract()


## test 5, example after Lemma 2.3
Phi5 = []
Phi5.append(  np.array([[1, -1, 1], [1, -1, 1]]) )
Phi5.append(  np.array([[-1], [-2]]) )
Phi5.append(  np.array([[1, -1]]) )
Phi5.append(  np.array([[0]]) )
nn2mv = NN2MV(Phi5)
nn2mv.extract()


## test 6, 2 fold hat function with shallow realization
Phi6 = []
Phi6.append(  np.array([[4], [4], [4],[4],[4]]) )
Phi6.append(  np.array([[0], [-1], [-2],[-3],[-4]]) )
Phi6.append(  np.array([[1, -2, 2, -2, 1]]) )
Phi6.append(  np.array([[0]]) )
nn2mv3 = NN2MV(Phi6)
nn2mv3.extract()


## test 7, two-dim example in Section 3 rho(x+y)-rho(y-x)-rho(x+y-1)
Phi7 = []
Phi7.append(  np.array([[1, 1], [-1, 1], [1, 1]]) )
Phi7.append(  np.array([[0], [0], [-1]]) )
Phi7.append(  np.array([[1, -1, -1]]) )
Phi7.append(  np.array([[0]]) )
nn2mv3 = NN2MV(Phi7)
nn2mv3.extract()


## test 8, two-dim example in Section 3 rho(x+y)-rho(y-x)-rho(x+y-1), after augmentation
Phi8 = []
Phi8.append(  np.array([[1, 1], [-1, 1], [1, 1]]) )
Phi8.append(  np.array([[0], [0], [-1]]) )
Phi8.append(  np.array([[1, -1, -1], [-1, 1, 1]]) )
Phi8.append(  np.array([[0], [0]]) )
Phi8.append(  np.array([[1, -1]]) )
Phi8.append(  np.array([[0]]) )
nn2mv3 = NN2MV(Phi8)
nn2mv3.extract()

#
# ## test 2, 2-fold hat function
#
# Phi2 = []
# Phi2.append(  np.array([[2],[2], [2]]) )
# Phi2.append(  np.array([[0], [-1], [-2]]) )
# Phi2.append(np.array([[2, -4, 2],
#                      [2, -4, 2],
#                         [2, -4, 2]]))
# Phi2.append(np.array([[0], [-1], [-2]]))
# Phi2.append(  np.array([[1, -2, 1]]) )
# Phi2.append(  np.array([[0]]) )
#
# nn2mv2 = NN2MV(Phi2)
# nn2mv2.extract()
#
#
# ## test 3, 2 fold hat function
# Phi3 = []
# Phi3.append(  np.array([[4], [4], [4],[4],[4]]) )
# Phi3.append(  np.array([[0], [-1], [-2],[-3],[-4]]) )
# Phi3.append(  np.array([[1, -2, 2, -2, 1]]) )
# Phi3.append(  np.array([[0]]) )
#
# nn2mv3 = NN2MV(Phi3)
# nn2mv3.extract()