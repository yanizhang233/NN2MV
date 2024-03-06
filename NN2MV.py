import numpy as np
from nonlinearity import sigma, relu
from helper import removeRedundant
from config import Symbols


class NN2MV:
    def __init__(self, Phi):
        ''' Initialize the class
        :param Phi:= (W1, b1, W2, b2, ..., Wn, bn), a list of numpy arrays
        '''
        self.nLayers = len(Phi) // 2 ## number of layers
        self.dims = [Phi[2 * i].shape[0] for i in range(self.nLayers)] ## number of neurons in each layer
        self.dims.insert(0, Phi[0].shape[1]) ## include the input layer
        self.mapping = Phi ## affine maps of all layers


    def transform(self):
        ''' Transform Phi into a CReLU-network
        new affine mappingss stored in self.affineMaps
        self.dims updated accordingly
        '''
        self.affineMaps = {}
        for l in range(self.nLayers-1): ## for each layer
            W = np.empty(shape=(0, self.dims[l]), dtype='int32')
            B = np.empty(shape=(0, 1), dtype='int32')
            W_next = np.empty(shape=(self.dims[l+1], 0), dtype='int32')
            for i in range(self.dims[l+1]): ## for each neuron
                w = self.mapping[2 * l][i]
                b = self.mapping[2 * l + 1][i]
                u = int(np.ceil(np.sum(w [w > 0]) + b))
                if u > 1:
                    W = np.append(W, np.tile(w, (u, 1)), axis = 0)
                    B = np.append(B, np.arange(b, b - u, -1).reshape(-1, 1), axis = 0)
                    W_next = np.append(W_next,
                        np.tile(np.eye(1, self.dims[l+1], k=i).reshape(-1, 1).astype('int32'),
                                (1, u)), axis = 1
                                )
                elif u > 0:
                    W = np.append(W, w.reshape(1, -1), axis=0)
                    B = np.append(B, b.reshape(1, -1), axis=0)
                    W_next = np.append(W_next, np.eye(1, self.dims[l+1], k=i).reshape(-1, 1).astype('int32'), axis=1)

            W, B, W_next = removeRedundant(W, B, W_next) ## check redundant sigma neurons

            self.affineMaps['W' + str(l+1)] = W
            self.affineMaps['B' + str(l+1)] = B
            self.mapping[2 * l] = W
            self.mapping[2 * l + 1] = B
            self.mapping[2 * l + 2] = np.matmul(self.mapping[2 * l + 2], W_next)
            self.dims[l + 1] = W.shape[0]

            if l == self.nLayers - 2:
                self.affineMaps['W' + str(l + 2)] = self.mapping[2 * l + 2]
                self.affineMaps['B' + str(l + 2)] = self.mapping[2 * l + 3]


    def sigmaConstruct(self, w, b):
        ''' Apply Lemma 3.4 in (ICML, 2024) to construct an MV term associated with the neuron sigma(wx+b)
        :param w: coefficients
        :param b: bias
        :return: extracted MV term
        '''
        if np.all(w==0):
            return str(int(sigma(b)))
        elif np.all(w<=0):
            return f"{Symbols.NOT} ({self.sigmaConstruct(-1*w, -1*b+1)})"
            # return '¬ ' + '(' + str(self.sigmaConstruct(-1*w, -1*b+1)) + ')'
        else:
            idx = np.where(w>0)[0][0]
            wl, wr = w.copy(), w.copy()
            wl[idx] -= 1
            wr[idx] -= 1
            left = self.sigmaConstruct(wl, b)
            right = self.sigmaConstruct(wr, b+1)
            if right == '0' or right == '¬ (1)':
                return '0'
            if left == '1' or left == '¬ (0)':
                return str(right)
            if right == '1' or right == '¬ (0)':
                if left == '0' or left == '¬ (1)':
                    return f"x{idx+1}"
                    # return 'x' +str(idx+1)
                else:
                    return f"({left} {Symbols.OR} x{idx+1})"
                    # return '(' + str(left) + ' ⊕ ' + 'x' +str(idx+1)+')'
            if left == '0' or left == '¬ (1)':
                if right == '1' or right == '¬ (0)':
                    return f"x{idx+1}"
                    # return 'x' +str(idx+1)
                else:
                    return f"(x{idx+1} {Symbols.AND} {right})"
                    # return '(' +'x' +str(idx+1) + ' ⊙ ' + str(right)+ ')'
            #return '(' + '(' + str(left) + ' ⊕ ' + 'x' +str(idx+1)+')' + ' ⊙ '+ str(right) + ')'
            return f"(({left} {Symbols.OR} x{idx+1}) {Symbols.AND} {right})"
            #return f"(({left} {Symbols.AND} x {idx}  1)) {Symbols.OR} {right} )"



    def construct(self):
        ''' Construct an MV term for each neuron of the CReLU network, results stored in self.MV_terms
        Print in console MV terms for each neuron in each layer
        '''
        self.MV_terms = {}
        for l in range(self.nLayers):
            print('------------------' +'Layer' + str(l + 1) + '------------------')
            self.MV_terms['Layer' + str(l + 1)] = []
            for i in range(self.dims[l + 1]):
                w = self.affineMaps['W' + str(l+1)][i]
                b = self.affineMaps['B' + str(l+1)][i]
                MV_term = self.sigmaConstruct(w, b)
                self.MV_terms['Layer' + str(l + 1)].append(MV_term)
                print('Node' + str(i+1) + ':   ' + MV_term)

    def compose(self):
        ''' Compose the MV terms layer by layer
        final result stored in self.MV_terms_composed['Layer' + str(self.nLayers)]
        Print out in Console the overall MV term
        '''
        self.MV_terms_composed = self.MV_terms.copy()
        for l in range(self.nLayers-1):
            for j in range(self.dims[l+2]):
                for i in range(self.dims[l+1]):
                    s1 = 'x' + str(i+1)
                    toReplace = self.MV_terms_composed['Layer' + str(l + 2)][j]
                    self.MV_terms_composed['Layer' + str(l + 2)][j] = toReplace.replace(s1, 's'+str(i+1))

            for j in range(self.dims[l+2]):
                for i in range(self.dims[l+1]):
                    s1 = 's' + str(i+1)
                    toReplace = self.MV_terms_composed['Layer' + str(l + 2)][j]
                    self.MV_terms_composed['Layer' + str(l + 2)][j] = toReplace.replace(s1, self.MV_terms_composed['Layer' + str(l + 1)][i])
        print('------------------overall MV term------------------')
        print(self.MV_terms_composed['Layer' + str(self.nLayers)][0])


    def extract(self):
        ## step 1: transform into a CReLU-network
        self.transform()
        ## step 2: construct an MV-term for each neuron of the transformed network
        self.construct()
        ## step 3: compose the MV-terms
        self.compose()



class NN2MVRational:
    def __init__(self, Phi, s):
        ''' Initialize the class
        :param
        Phi:= (W1, b1, W2, b2, ..., Wn, bn), a list of numpy arrays
        s: least common multiple of denominator of all coefficients, the precision of approximation, dtype=integer
        '''
        self.s = s
        self.nLayers = len(Phi) // 2 ## number of layers
        self.dims = [Phi[2 * i].shape[0] for i in range(self.nLayers)] ## number of neurons in each layer
        self.dims.insert(0, Phi[0].shape[1]) ## include the input layer
        ##
        for l in range(len(Phi)):
            Phi[l] = ((s * Phi[l]).round())/s
        self.mapping = Phi ## affine maps of all layers


    def transform(self):
        ''' Transform Phi into a CReLU-network
        new affine mappingss stored in self.affineMaps
        self.dims updated accordingly
        '''
        self.affineMaps = {}
        for l in range(self.nLayers-1): ## for each layer
            W = np.empty(shape=(0, self.dims[l]), dtype='float64')
            B = np.empty(shape=(0, 1), dtype='float64')
            W_next = np.empty(shape=(self.dims[l+1], 0), dtype='float64')
            for i in range(self.dims[l+1]): ## for each neuron
                w = self.mapping[2 * l][i]
                b = self.mapping[2 * l + 1][i]
                u = int(np.ceil(np.sum(w [w > 0]) + b))
                if u > 1:
                    W = np.append(W, np.tile(w, (u, 1)), axis = 0)
                    B = np.append(B, np.arange(b, b - u, -1).reshape(-1, 1), axis = 0)
                    W_next = np.append(W_next,
                        np.tile(np.eye(1, self.dims[l+1], k=i).reshape(-1, 1).astype('int32'),
                                (1, u)), axis = 1
                                )
                elif u > 0:
                    W = np.append(W, w.reshape(1, -1), axis=0)
                    B = np.append(B, b.reshape(1, -1), axis=0)
                    W_next = np.append(W_next, np.eye(1, self.dims[l+1], k=i).reshape(-1, 1).astype('int32'), axis=1)

            W, B, W_next = removeRedundant(W, B, W_next) ## check redundant sigma neurons

            self.affineMaps['W' + str(l+1)] = W
            self.affineMaps['B' + str(l+1)] = B
            self.mapping[2 * l] = W
            self.mapping[2 * l + 1] = B
            self.mapping[2 * l + 2] = np.matmul(self.mapping[2 * l + 2], W_next)
            self.dims[l + 1] = W.shape[0]

            if l == self.nLayers - 2:
                self.affineMaps['W' + str(l + 2)] = self.mapping[2 * l + 2]
                self.affineMaps['B' + str(l + 2)] = self.mapping[2 * l + 3]


    def sigmaConstruct(self, w, b):
        ''' Apply Lemma 3.4 in (ICML, 2024) to construct an MV term associated with the neuron sigma(wx+b)
        :param w: coefficients
        :param b: bias
        :return: extracted MV term
        '''
        if np.all(w==0):
            if b <= 0 or b >= 1:
                return str(int(sigma(b)))
            else:
                return f"({(' ⊙ ').join(['δ' + '_' + str(self.s) + ' 1'] * int(b * self.s))})"
                #return '(' + (' ⊙ ').join(['δ' + '_' + str(self.s) + ' 1'] * int(b * self.s)) + ')'
        elif np.all(w<=0):
            return f"{Symbols.NOT} ({self.sigmaConstruct(-1*w, -1*b+1)})"
            # return '¬ ' + '(' + str(self.sigmaConstruct(-1*w, -1*b+1)) + ')'
        elif np.all(w < 1):
            idx = np.where(w > 0)[0][0]
            wl, wr = w.copy(), w.copy()
            wl[idx] = 0
            wr[idx] = 0
            left = self.sigmaConstruct(wl, b)
            right = self.sigmaConstruct(wr, b + 1)
            if right == '0' or right == '¬ (1)':
                return '0'
            if left == '1' or left == '¬ (0)':
                return str(right)
            if right == '1' or right == '¬ (0)':
                if left == '0' or left == '¬ (1)':
                    return ' ⊕ '.join(['δ' + '_' + str(self.s) + 'x' + str(idx + 1)] * int(w[idx] * self.s))
                else:
                    return f"({left} {Symbols.OR} {' ⊕ '.join(['δ' + '_' + str(self.s) + 'x' + str(idx + 1)] * int(w[idx] * self.s))})"
                    # return '(' + str(left) + ' ⊕ ' +  ' ⊕ '.join(['δ' + '_' + str(self.s) + 'x' + str(idx + 1)] * int(w[idx] * self.s)) + ')'
            if left == '0' or left == '¬ (1)':
                if right == '1' or right == '¬ (0)':
                    return ' ⊕ '.join(['δ' + '_' + str(self.s) + 'x' + str(idx + 1)] * int(w[idx] * self.s))
                else:
                    return f"(({' ⊕ '.join(['δ' + '_' + str(self.s) + 'x' + str(idx + 1)] * int(w[idx] * self.s))}) {Symbols.AND} {right})"
                    # return '(' + '('+ ' ⊕ '.join(['δ' + '_' + str(self.s) + 'x' + str(idx + 1)] * int(w[idx] * self.s)) +')' + ' ⊙ ' + str(right)+ ')'
            return f"(({left} {Symbols.OR} x{idx+1}) {Symbols.AND} {right})"
            #return '(' + '(' + str(left) + ' ⊕ ' + 'x' +str(idx+1)+')' + ' ⊙ '+ str(right) + ')'


        else:
            idx = np.where(w>0)[0][0]
            wl, wr = w.copy(), w.copy()
            wl[idx] -= 1
            wr[idx] -= 1
            left = self.sigmaConstruct(wl, b)
            right = self.sigmaConstruct(wr, b+1)
            if right == '0' or right == '¬ (1)':
                return '0'
            if left == '1' or left == '¬ (0)':
                return str(right)
            if right == '1' or right == '¬ (0)':
                if left == '0' or left == '¬ (1)':
                    return 'x' +str(idx+1)
                else:
                    return f"({left} {Symbols.OR} x{idx+1})"
                    #return '(' + str(left) + ' ⊕ ' + 'x' +str(idx+1)+')'
            if left == '0' or left == '¬ (1)':
                if right == '1' or right == '¬ (0)':
                    return 'x' +str(idx+1)
                else:
                    return f"(x{idx+1} {Symbols.AND} {right})"
                    # return '(' +'x' +str(idx+1) + ' ⊙ ' + str(right)+ ')'
            return f"(({left} {Symbols.OR} x{idx+1}) {Symbols.AND} {right})"
            # return '(' + '(' + str(left) + ' ⊕ ' + 'x' +str(idx+1)+')' + ' ⊙ '+ str(right) + ')'

    def construct(self):
        ''' Construct an MV term for each neuron of the CReLU network, results stored in self.MV_terms
        Print in console MV terms for each neuron in each layer
        '''
        self.MV_terms = {}
        for l in range(self.nLayers):
            print('------------------' +'Layer' + str(l + 1) + '------------------')
            self.MV_terms['Layer' + str(l + 1)] = []
            for i in range(self.dims[l + 1]):
                w = self.affineMaps['W' + str(l+1)][i]
                b = self.affineMaps['B' + str(l+1)][i]
                MV_term = self.sigmaConstruct(w, b)
                self.MV_terms['Layer' + str(l + 1)].append(MV_term)
                print('Node' + str(i+1) + ':   ' + MV_term)

    def compose(self):
        ''' Compose the MV terms layer by layer
        final result stored in self.MV_terms_composed['Layer' + str(self.nLayers)]
        Print out in Console the overall MV term
        '''
        self.MV_terms_composed = self.MV_terms.copy()
        for l in range(self.nLayers-1):
            for j in range(self.dims[l+2]):
                for i in range(self.dims[l+1]):
                    s1 = 'x' + str(i+1)
                    toReplace = self.MV_terms_composed['Layer' + str(l + 2)][j]
                    self.MV_terms_composed['Layer' + str(l + 2)][j] = toReplace.replace(s1, 's'+str(i+1))

            for j in range(self.dims[l+2]):
                for i in range(self.dims[l+1]):
                    s1 = 's' + str(i+1)
                    toReplace = self.MV_terms_composed['Layer' + str(l + 2)][j]
                    self.MV_terms_composed['Layer' + str(l + 2)][j] = toReplace.replace(s1, self.MV_terms_composed['Layer' + str(l + 1)][i])
        print('------------------overall MV term------------------')
        print(self.MV_terms_composed['Layer' + str(self.nLayers)][0])


    def extract(self):
        ## step 1: transform into a CReLU-network
        self.transform()
        ## step 2: construct an MV-term for each neuron of the transformed network
        self.construct()
        ## step 3: compose the MV-terms
        self.compose()



class NN2MVReal:
    def __init__(self, Phi):
        ''' Initialize the class
        :param Phi:= (W1, b1, W2, b2, ..., Wn, bn), a list of numpy arrays
        '''
        self.nLayers = len(Phi) // 2 ## number of layers
        self.dims = [Phi[2 * i].shape[0] for i in range(self.nLayers)] ## number of neurons in each layer
        self.dims.insert(0, Phi[0].shape[1]) ## include the input layer
        self.mapping = Phi ## affine maps of all layers


    def transform(self):
        ''' Transform Phi into a CReLU-network
        new affine mappingss stored in self.affineMaps
        self.dims updated accordingly
        '''
        self.affineMaps = {}
        for l in range(self.nLayers-1): ## for each layer
            W = np.empty(shape=(0, self.dims[l]), dtype='int32')
            B = np.empty(shape=(0, 1), dtype='int32')
            W_next = np.empty(shape=(self.dims[l+1], 0), dtype='int32')
            for i in range(self.dims[l+1]): ## for each neuron
                w = self.mapping[2 * l][i]
                b = self.mapping[2 * l + 1][i]
                u = int(np.ceil(np.sum(w [w > 0]) + b))
                if u > 1:
                    W = np.append(W, np.tile(w, (u, 1)), axis = 0)
                    B = np.append(B, np.arange(b, b - u, -1).reshape(-1, 1), axis = 0)
                    W_next = np.append(W_next,
                        np.tile(np.eye(1, self.dims[l+1], k=i).reshape(-1, 1).astype('int32'),
                                (1, u)), axis = 1
                                )
                elif u > 0:
                    W = np.append(W, w.reshape(1, -1), axis=0)
                    B = np.append(B, b.reshape(1, -1), axis=0)
                    W_next = np.append(W_next, np.eye(1, self.dims[l+1], k=i).reshape(-1, 1).astype('int32'), axis=1)

            W, B, W_next = removeRedundant(W, B, W_next) ## check redundant sigma neurons

            self.affineMaps['W' + str(l+1)] = W
            self.affineMaps['B' + str(l+1)] = B
            self.mapping[2 * l] = W
            self.mapping[2 * l + 1] = B
            self.mapping[2 * l + 2] = np.matmul(self.mapping[2 * l + 2], W_next)
            self.dims[l + 1] = W.shape[0]

            if l == self.nLayers - 2:
                self.affineMaps['W' + str(l + 2)] = self.mapping[2 * l + 2]
                self.affineMaps['B' + str(l + 2)] = self.mapping[2 * l + 3]


    def sigmaConstruct(self, w, b):
        ''' Apply Lemma 3.4 in (ICML, 2024) to construct an MV term associated with the neuron sigma(wx+b)
        :param w: coefficients
        :param b: bias
        :return: extracted MV term
        '''
        if np.all(w==0):
            if b[0] <= 0:
                return str('0')
            elif b[0] >= 1:
                return str('1')
            else:
                return str(sigma(b[0]))
        elif np.all(w<=0):
            return f"{Symbols.NOT} ({self.sigmaConstruct(-1*w, -1*b+1)})"
            #return '¬ ' + '(' + str(self.sigmaConstruct(-1*w, -1*b+1)) + ')'
        elif np.all(w < 1):
            idx = np.where(w > 0)[0][0]
            wl, wr = w.copy(), w.copy()
            wl[idx] = 0
            wr[idx] = 0
            left = self.sigmaConstruct(wl, b)
            right = self.sigmaConstruct(wr, b + 1)
            if right == '0' or right == '¬ (1)':
                return '0'
            if left == '1' or left == '¬ (0)':
                return str(right)
            if right == '1' or right == '¬ (0)':
                if left == '0' or left == '¬ (1)':
                    return f"{w[idx]}x{idx+1}"
                    # return str(w[idx]) + 'x' + str(idx + 1)
                else:
                    return f"({left} {Symbols.OR} ({w[idx]}x{idx+1}))"
                    # return '(' + str(left) + ' ⊕ ' + '(' + str(w[idx]) + 'x' + str(idx + 1) + ')' + ')'
            if left == '0' or left == '¬ (1)':
                if right == '1' or right == '¬ (0)':
                    return f"{w[idx]}x{idx+1}"
                    # return str(w[idx]) + 'x' + str(idx + 1)
                else:
                    return f"(({w[idx]}x{idx+1}) {Symbols.AND} {right})"
                    # return '(' + '(' + str(w[idx]) + 'x' + str(idx + 1) + ')' + ' ⊙ ' + str(right) + ')'
            return f"(({left} {Symbols.OR} x{idx+1}) {Symbols.AND} {right})"
            # return '(' + '(' + str(left) + ' ⊕ ' + 'x' + str(idx + 1) + ')' + ' ⊙ ' + str(right) + ')'

        else:
            idx = np.where(w>0)[0][0]
            wl, wr = w.copy(), w.copy()
            wl[idx] -= 1
            wr[idx] -= 1
            left = self.sigmaConstruct(wl, b)
            right = self.sigmaConstruct(wr, b+1)
            if right == '0' or right == '¬ (1)':
                return '0'
            if left == '1' or left == '¬ (0)':
                return str(right)
            if right == '1' or right == '¬ (0)':
                if left == '0' or left == '¬ (1)':
                    return 'x' +str(idx+1)
                else:
                    return f"({left} {Symbols.OR} x{idx+1})"
                    # return '(' + str(left) + ' ⊕ ' + 'x' +str(idx+1)+')'
            if left == '0' or left == '¬ (1)':
                if right == '1' or right == '¬ (0)':
                    return 'x' +str(idx+1)
                else:
                    return f"(x{idx+1} {Symbols.AND} {right})"
                    # return '(' +'x' +str(idx+1) + ' ⊙ ' + str(right)+ ')'
            return f"(({left} {Symbols.OR} x{idx+1}) {Symbols.AND} {right})"
            # return '(' + '(' + str(left) + ' ⊕ ' + 'x' +str(idx+1)+')' + ' ⊙ '+ str(right) + ')'

    def construct(self):
        ''' Construct an MV term for each neuron of the CReLU network, results stored in self.MV_terms
        Print in console MV terms for each neuron in each layer
        '''
        self.MV_terms = {}
        for l in range(self.nLayers):
            print('------------------' +'Layer' + str(l + 1) + '------------------')
            self.MV_terms['Layer' + str(l + 1)] = []
            for i in range(self.dims[l + 1]):
                w = self.affineMaps['W' + str(l+1)][i]
                b = self.affineMaps['B' + str(l+1)][i]
                MV_term = self.sigmaConstruct(w, b)
                self.MV_terms['Layer' + str(l + 1)].append(MV_term)
                print('Node' + str(i+1) + ':   ' + MV_term)

    def compose(self):
        ''' Compose the MV terms layer by layer
        final result stored in self.MV_terms_composed['Layer' + str(self.nLayers)]
        Print out in Console the overall MV term
        '''
        self.MV_terms_composed = self.MV_terms.copy()
        for l in range(self.nLayers-1):
            for j in range(self.dims[l+2]):
                for i in range(self.dims[l+1]):
                    s1 = 'x' + str(i+1)
                    toReplace = self.MV_terms_composed['Layer' + str(l + 2)][j]
                    self.MV_terms_composed['Layer' + str(l + 2)][j] = toReplace.replace(s1, 's'+str(i+1))

            for j in range(self.dims[l+2]):
                for i in range(self.dims[l+1]):
                    s1 = 's' + str(i+1)
                    toReplace = self.MV_terms_composed['Layer' + str(l + 2)][j]
                    self.MV_terms_composed['Layer' + str(l + 2)][j] = toReplace.replace(s1, self.MV_terms_composed['Layer' + str(l + 1)][i])
        print('------------------overall MV term------------------')
        print(self.MV_terms_composed['Layer' + str(self.nLayers)][0])


    def extract(self):
        ## step 1: transform into a CReLU-network
        self.transform()
        ## step 2: construct an MV-term for each neuron of the transformed network
        self.construct()
        ## step 3: compose the MV-terms
        self.compose()