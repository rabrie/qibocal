import sys
import tqdm
sys.path.append('../../src/mGST')

from additional_fns import *
from low_level_jit import *
from algorithm import * 
from compatibility import *
from optimization import *


######################################	Parameters
pdim = 2   # physical dimension
r = pdim**2   # rank of the gate superoperators 

l = 7  # maximum number of gates in each measurement sequence
d = 3  # number of gates
rK = 4   # rank of the model estimate
n_povm = 2   # number of POVM-elements
bsize = 50   # The batch size on which the optimization is started

N = 100 #Number of sequences
meas_samples = 1e3 #Number of samples per sequence



######################################  Import exp. data..
# J_rand = np.array(random.sample(range(d**l), N))   # generate random numbers between 0 and $d^l - 1$
# J = np.array([local_basis(ind,d,l) for ind in J_rand])   # turn random numbers into gate instructions    
# print(J)

######################################  Run mGST
t = time.time()
K,X,E,rho,res_list,_ = run_mGST(y,J,l,d,r,rK, n_povm, bsize, meas_samples, method = 'SFN',
                     max_inits = 10, max_iter = 200, final_iter = 100, 
                     target_rel_prec = 1e-4)
#     plt.semilogy(res_list)   # plot the objective function over the iterations
print('MVE:', MVE(X_true,E_true,rho_true,X,E,rho,d,l, n_povm)[0])   # output the final mean variation error
print('Time:', time.time()-t)
		