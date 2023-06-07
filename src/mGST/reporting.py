from mGST import compatibility,low_level_jit,additional_fns
from pygsti.report import reportables as rptbl
from pygsti.algorithms import gaugeopt_to_target
from pygsti.models import gaugegroup
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd

from matplotlib import rcParams
import matplotlib.ticker as ticker

# rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm,amsmath,amssymb,lmodern}')
plt.rcParams.update({'font.family':'computer-modern'})


SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def min_spectral_distance(X1,X2):
    r = X1.shape[0]
    eigs = la.eig(X1)[0]
    eigs_t = la.eig(X2)[0]
    cost_matrix = np.array([[np.abs(eigs[i] - eigs_t[j]) for i in range(r)] for j in range(r)])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    normalization = np.abs(eigs).sum()
    return cost_matrix[row_ind,col_ind].sum()/normalization


def MVE_data(X, E, rho, J, y):
    m = y.shape[1]
    n_povm = y.shape[0]
    dist: float = 0
    max_dist: float = 0
    curr: float = 0
    for i in range(m):
        j = J[i]
        C = low_level_jit.contract(X, j)
        curr = 0
        for k in range(n_povm):
            y_model = E[k].conj()@C@rho
            curr += np.abs(y_model - y[k, i])
        curr = curr/2
        dist += curr
        if curr > max_dist:
            max_dist = curr
    return dist/m, max_dist


def gauge_opt(X, E, rho, target_mdl, weights):
    pdim = int(np.sqrt(rho.shape[0]))
    mdl = compatibility.arrays_to_pygsti_model(X,E,rho, basis = 'std')
    X_t,E_t,rho_t = compatibility.pygsti_model_to_arrays(target_mdl,basis = 'std')
    target_mdl = compatibility.arrays_to_pygsti_model(X_t,E_t,rho_t, basis = 'std') #For consistent gate labels

    gauge_optimized_mdl = gaugeopt_to_target(mdl, 
                target_mdl,gauge_group = gaugegroup.UnitaryGaugeGroup(target_mdl.state_space, basis = 'pp'),
                item_weights=weights)
    return compatibility.pygsti_model_to_arrays(gauge_optimized_mdl,basis = 'std')  

def report(X, E, rho, J, y, target_mdl, gate_labels):
    pdim = int(np.sqrt(rho.shape[0]))
    X_t,E_t,rho_t = compatibility.pygsti_model_to_arrays(target_mdl,basis = 'std')
    target_mdl = compatibility.arrays_to_pygsti_model(X_t,E_t,rho_t, basis = 'std') #For consistent gate labels

    gauge_optimized_mdl = compatibility.arrays_to_pygsti_model(X,E,rho, basis = 'std')
        
    final_objf = low_level_jit.objf(X,E,rho,J,y)
    MVE = MVE_data(X,E,rho,J,y)[0]
    MVE_target = MVE_data(X_t,E_t,rho_t,J,y)[0]
    povm_td = rptbl.povm_jtrace_diff(target_mdl, gauge_optimized_mdl, 'Mdefault')
    rho_td = la.norm(rho.reshape((pdim,pdim))-rho_t.reshape((pdim,pdim)),ord = 'nuc')/2
    F_avg = compatibility.average_gate_fidelities(gauge_optimized_mdl,target_mdl,pdim, basis_string = 'pp')
    DD = compatibility.diamond_dists(gauge_optimized_mdl,target_mdl,pdim, basis_string = 'pp')
    min_spectral_dists = [min_spectral_distance(X[i],X_t[i]) for i in range(X.shape[0])]
    

    df_g = pd.DataFrame({
        "F_avg":F_avg,
        "Diamond distances": DD,
        "Min. Spectral distances": min_spectral_dists
    })
    df_o = pd.DataFrame({
        "Final cost function value": final_objf,
        "Mean total variation dist. to data": MVE,
        "Mean total variation dist. target to data": MVE_target,
        "POVM - Choi map trace distance": povm_td,
        "State - Trace distance": rho_td,  
    }, index = [0])
    df_g.rename(index=gate_labels, inplace = True)
    df_o.rename(index={0: ""}, inplace = True)
    
    s_g = df_g.style.format(precision=5, thousands=".", decimal=",")
    s_o = df_o.style
    
    s_g.set_table_styles([
    {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    {'selector': 'td', 'props': 'text-align: center'},
    ], overwrite=False)
    s_o.set_table_styles([
    {'selector': 'th.col_heading', 'props': 'text-align: center;'},
    {'selector': 'th.col_heading.level0', 'props': 'font-size: 1em;'},
    {'selector': 'td', 'props': 'text-align: center'},
    ], overwrite=False)
    return df_g, df_o, s_g, s_o

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
def plot_mat(mat1, mat2):
    dim = mat1.shape[0]
    fig, axes = plt.subplots(ncols=2, nrows = 1,gridspec_kw={"width_ratios":[1,1]}, sharex=True)
    plt.rc('image', cmap='RdBu')
    ax = axes[0]
    im0 = ax.imshow(np.real(mat1), vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
    ax.set_xticks(np.arange(dim))
    ax.set_xticklabels(np.arange(dim)+1)
    ax.set_yticks(np.arange(dim))
    ax.set_yticklabels(np.arange(dim)+1)
    ax.grid(visible = 'True', alpha = 0.4)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax = axes[1]
    im1 = ax.imshow(np.real(mat2), vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
    ax.set_xticks(np.arange(dim))
    ax.set_xticklabels(np.arange(dim)+1)
    ax.set_yticks(np.arange(dim))
    ax.set_yticklabels(np.arange(dim)+1)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.grid(visible = 'True', alpha = 0.4)
    axes[0].set_title(r'GST result')
    axes[1].set_title(r'Ideal gate')

    # cax = fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0-0.05,0.02,ax.get_position().height])
    #fig.colorbar(im1, cax=cax)
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), pad = 0.1)
    cbar.ax.set_ylabel(r'Matrix \, entry $\times 10$', labelpad = 5, rotation=90)


    fig.subplots_adjust(left = 0.05, right = .7, top = 1, bottom = -.1)

    set_size(np.sqrt(3*dim),np.sqrt(dim)*1.2)

    plt.show()
    return im0, im1
    
def plot_spam(rho, E):
    r = rho.shape[0]
    n_povm = E.shape[0]
    fig, axes = plt.subplots(ncols = 1, nrows=n_povm+1, sharex=True)
    plt.rc('image', cmap='RdBu')
    
    ax = axes[0]
    im0 = ax.imshow(np.real(rho).reshape(1,r), vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
    ax.set_xticks(np.arange(r))
    ax.set_title(r'$\rho$')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    
    for i in range(n_povm): 
        ax = axes[1+i]
        ax.imshow(np.real(E[i].reshape(1,r)), vmin = -1, vmax = 1) #change_basis(S_true_maps[0],'std','pp')
        ax.set_xticks(np.arange(r))
        ax.set_xticklabels(np.arange(r)+1)
        ax.set_title(r'POVM-Element %i'%(i+1))
        ax.yaxis.set_major_locator(ticker.NullLocator())

#     cax = fig.add_axes([axes[0].get_position().x1+0.05,ax.get_position().y0-0.05,0.02,10*ax.get_position().height])
#     fig.colorbar(im0, cax=cax)
    
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), pad = 0.1)
    cbar.ax.set_ylabel(r'Pauli basis coefficient', labelpad = 5, rotation=90)

    #fig.subplots_adjust(left = 0.05, right = .7, top = 1, bottom = -.1)
    set_size(4,3)
    plt.show()
    return im0
