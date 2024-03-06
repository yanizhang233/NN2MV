## NN2MV  
NN2MV is a package for extracting logical formuale in MV logic from ReLU neural networks.  
## How to use  
```python  
from NN2MV import NN2MV  
import numpy as np  
  
# Declare the affine maps of each layer.  
Phi = []  
Phi.append(  np.array([[2],[2], [2]]) ) Phi.append(  np.array([[0], [-1], [-2]]) )  
Phi.append(  np.array([[1, -2, 1]]) )  
Phi.append(  np.array([[0]]) )  
  
# Initialize the NN2MV object.  
nn2mv = NN2MV(Phi)  
  
# Call the extract method. Results printed in console.  
nn2mv.extract()  
```  
  
  
  
  
## Modules   
  
### nonlinearity  
This module contains two nonlinearities needed for the extraction procedure, namely the rectified linear unit (ReLU) given by $\rho(x) = \max\{0,x\}$ and the clipped rectified linear unit (CReLU) nonlinearty given by $\sigma(x) = \min\{1, \max\{0,x\}\}$  
  
### helper  
This helper contains helper functions. The most important one is 'removeRedundant', which removes redundance $\sigma$-neurons in Step 1 and help simplify the overall MV term.   
  
### NN2MV  
The class NN2MV implements the proposed algorithm.  
  
### tests  
This module contains three simple examples of using NN2MV to extract logical formualae from ReLU neural networks.   The first example is a ReLU network that realizes the hat function $g$ in Figure 2 in (ICML, 2024). The second and third example are ReLU networks that realize the $2$-fold hat function $g_2$ in Figure 2 in (ICML, 2024).
  