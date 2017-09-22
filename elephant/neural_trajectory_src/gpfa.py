import time

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.optimize as optimize
from sklearn.decomposition import FactorAnalysis

# import util
from . import util


def learn_gp_params(seq, params, verbose=False):
    """Updates parameters of GP state model given neural trajectories.

    parameters
    ----------
    seq : numpy recarray
          data structure containing neural trajectories
    params : dict
             current GP state model parameters, which gives starting point
             for gradient optimization
    verbose : bool, optional
              specifies whether to display status messages (default: False)

    returns
    -------
    param_opt : ndarray
                updated GP state model parameter
    """
    cov_type = params['covType']
    learn_gp_noise = params['notes']['learnGPNoise']

    if cov_type == 'rbf':
        param_name = 'gamma'
        fname = 'util.grad_betgam'
    elif cov_type == 'tri':
        raise ValueError("learnKernelParams with covType = 'tri' is not supported.")
    elif cov_type == 'logexp':
        raise ValueError("learnKernelParams with covType = 'logexp' is not supported.")
    else:
        raise ValueError("covType must be either 'rbf', 'tri' or 'logexp'.")

    if learn_gp_noise:
        raise ValueError("learnGPNoise is not supported.")

    param_init = params[param_name]
    param_opt = {param_name: np.empty_like(param_init)}

    x_dim = param_init.shape[-1]
    precomp = util.make_precomp(seq, x_dim)

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        const = {}
        if not learn_gp_noise:
            const['eps'] = params['eps'][i]

        initp = np.log(param_init[i])
        res_opt = optimize.minimize(eval(fname), initp, args=(precomp[i], const), method='L-BFGS-B', jac=True)
        param_opt['gamma'][i] = np.exp(res_opt.x)

        if verbose:
            print('\n Converged p; xDim:{}, p:{}'.format(i, res_opt.x))

    return param_opt


def exact_inference_with_ll(seq, params, get_ll=True):
    """Extracts latent trajectories from neural data given GPFA model parameters.

    Parameters
    ----------
    seq : numpy recarray
          Input data structure, whose n-th element (corresponding to the n-th experimental trial) has fields:
              y : ndarray of shape (#units, #bins)
                  neural data
              T : int
                  number of bins
    params : dict
             GPFA model parameters contained in fields:
                C : ndarray
                    FA factor loadings matrix
                d : ndarray
                    FA mean vector
                R : ndarray
                    FA noise covariance matrix
                gamma : ndarray
                        GP timescale
                eps : ndarray
                      GP noise variance
    get_ll : bool, optional
             specifies whether to compute data log likelihood (default: True)

    Returns
    -------
    seq_lat : numpy recarray
              a copy of the input data structure, augmented by new fields:
                  xsm : ndarray of shape (#latent_vars x #bins)
                        posterior mean of latent variables at each time bin
                  Vsm : ndarray of shape (#latent_vars, #latent_vars, #bins)
                        posterior covariance between latent variables at each timepoint
                  VsmGP : ndarray of shape (#bins, #bins, #latent_vars)
                          posterior covariance over time for each latent variable
    ll : float
         data log likelihood, np.nan is returned when `get_ll` is set False
    """
    y_dim, x_dim = params['C'].shape

    # copy the contents of the input data structure to output structure
    dtype_out = [(x, seq[x].dtype) for x in seq.dtype.names]
    dtype_out.extend([('xsm', np.object), ('Vsm', np.object), ('VsmGP', np.object)])
    seq_lat = np.empty(len(seq), dtype=dtype_out)
    for dtype_name in seq.dtype.names:
        seq_lat[dtype_name] = seq[dtype_name]

    # Precomputations
    if params['notes']['RforceDiagonal']:
        rinv = np.diag(1.0/np.diag(params['R']))
        logdet_r = (np.log(np.diag(params['R']))).sum()
    else:
        rinv = linalg.inv(params['R'])
        rinv = (rinv + rinv.T) / 2  # ensure symmetry
        logdet_r = util.logdet(params['R'])

    c_rinv = params['C'].T.dot(rinv)
    c_rinv_c = c_rinv.dot(params['C'])

    t_all = seq_lat['T']
    t_uniq = np.unique(t_all)
    ll = 0.

    # Overview:
    # - Outer loop on each element of Tu.
    # - For each element of Tu, find all trials with that length.
    # - Do inference and LL computation for all those trials together.
    for t in t_uniq:
        k_big, k_big_inv, logdet_k_big = util.make_k_big(params, t)
        k_big = sparse.csr_matrix(k_big)

        blah = [c_rinv_c for _ in range(t)]
        c_rinv_c_big = linalg.block_diag(*blah)  # (xDim*T) x (xDim*T)
        minv, logdet_m = util.inv_persymm(k_big_inv + c_rinv_c_big, x_dim,
                                          off_diag_sparse=True)

        # Note that posterior covariance does not depend on observations,
        # so can compute once for all trials with same T.
        # xDim x xDim posterior covariance for each timepoint
        vsm = np.full((x_dim, x_dim, t), np.nan)
        idx = np.arange(0, x_dim*t+1, x_dim)
        for i in range(t):
            vsm[:, :, i] = minv[idx[i]:idx[i+1], idx[i]:idx[i+1]]

        # T x T posterior covariance for each GP
        vsm_gp = np.full((t, t, x_dim), np.nan)
        for i in range(x_dim):
            vsm_gp[:, :, i] = minv[i::x_dim, i::x_dim]

        # Process all trials with length T
        n_list = np.where(t_all == t)[0]
        # dif is yDim x sum(T)
        dif = np.hstack(seq_lat[n_list]['y']) - params['d'][:, np.newaxis]
        # term1Mat is (xDim*T) x length(nList)
        term1_mat = c_rinv.dot(dif).reshape((x_dim*t, -1), order='F')

        # Compute blkProd = CRinvC_big * invM efficiently
        # blkProd is block persymmetric, so just compute top half
        t_half = np.int(np.ceil(t/2.0))
        blk_prod = np.zeros((x_dim*t_half, x_dim*t))
        idx = range(0, x_dim*t_half+1, x_dim)
        for i in range(t_half):
            blk_prod[idx[i]:idx[i+1], :] = c_rinv_c.dot(minv[idx[i]:idx[i+1], :])
        blk_prod = k_big[:x_dim*t_half, :].dot(
            util.fill_persymm(np.eye(x_dim * t_half, x_dim * t) - blk_prod, x_dim, t))
        # xsmMat is (xDim*T) x length(nList)
        xsm_mat = util.fill_persymm(blk_prod, x_dim, t).dot(term1_mat)

        for i, n in enumerate(n_list):
            seq_lat[n]['xsm'] = xsm_mat[:, i].reshape((x_dim, t), order='F')
            seq_lat[n]['Vsm'] = vsm
            seq_lat[n]['VsmGP'] = vsm_gp

        if get_ll:
            # Compute data likelihood
            val = -t*logdet_r - logdet_k_big - logdet_m \
                - y_dim*t*np.log(2*np.pi)
            ll = ll + len(n_list)*val - (rinv.dot(dif)*dif).sum() \
                + (term1_mat.T.dot(minv)*term1_mat.T).sum()

    if get_ll:
        ll /= 2
    else:
        ll = np.nan

    return seq_lat, ll


def em(params_init, seq, em_max_iters=500, tol=1.0E-8, min_var_frac=0.01,
       freq_ll=5, verbose=False):
    """Fits GPFA model parameters using expectation-maximization (EM) algorithm.

    Parameters
    ----------
    params_init : dict
                  GPFA model parameters at which EM algorithm is initialized
                      covType : {'rbf', 'tri', 'logexp'}
                                type of GP covariance
                      gamma : ndarray of shape (1, #latent_vars)
                              related to GP timescales by 'bin_width / sqrt(gamma)'
                      eps : ndarray of shape (1, #latent_vars)
                            GP noise variances
                      d : ndarray of shape (#units, 1)
                          observation mean
                      C : ndarray of shape (#units, #latent_vars)
                          mapping between the neuronal data space and the latent variable space
                      R : ndarray of shape (#units, #latent_vars)
                          observation noise covariance
    seq : numpy recarray
          training data structure, whose n-th entry (corresponding to the n-th experimental trial) has fields
              trialId :
                        unique trial identifier
              T : int
                  number of bins
              y : ndarray (yDim x T)
                  neural data
    em_max_iters : int, optional
                   number of EM iterations to run (default: 500)
    tol : float, optional
          stopping criterion for EM (default: 1e-8)
    min_var_frac : float, optional
                   fraction of overall data variance for each observed
                   dimension to set as the private variance floor.  This is
                   used to combat Heywood cases, where ML parameter learning
                   returns one or more zero private variances. (default: 0.01)
                   (See Martin & McDonald, Psychometrika, Dec 1975.)
    freq_ll : int, optional
              data likelihood is computed at every freq_ll EM iterations.
              freq_ll = 1 means that data likelihood is computed at every
              iteration. (default: 5)
    verbose : bool, optional
              specifies whether to display status messages (default: false)

    Returns
    -------
    params_est : dict
                 GPFA model parameter estimates returned by EM algorithm (same format as params_init)
    seq_lat : dict
              a copy of the training data structure, augmented by new fields
                  xsm : ndarray of shape (#latent_vars x #bins)
                        posterior mean of latent variables at each time bin
                  Vsm : ndarray of shape (#latent_vars, #latent_vars, #bins)
                        posterior covariance between latent variables at each timepoint
                  VsmGP : ndarray of shape (#bins, #bins, #latent_vars)
                          posterior covariance over time for each latent variable
    ll : list of floats
         data log likelihood after each EM iteration
    iter_time : list of floats
                computation time (in seconds) for each EM iteration
    """
    params = params_init
    t = seq['T']
    y_dim, x_dim = params['C'].shape
    lls = []
    ll = 0.0
    ll_base = ll
    iter_time = []
    var_floor = min_var_frac * np.diag(np.cov(np.hstack(seq['y'])))

    # Loop once for each iteration of EM algorithm
    for i in range(1, em_max_iters+1):
        if verbose:
            print()
        tic = time.time()

        print('EM iteration {0:3d} of {1}'.format(i, em_max_iters))
        if (np.fmod(i, freq_ll) == 0) or (i <= 2):
            get_ll = True
        else:
            get_ll = False

        # ==== E STEP =====
        if not np.isnan(ll):
            ll_old = ll
        seq_lat, ll = exact_inference_with_ll(seq, params, get_ll=get_ll)
        lls.append(ll)

        # ==== M STEP ====
        sum_p_auto = np.zeros((x_dim, x_dim))
        for seq_lat_n in seq_lat:
            sum_p_auto += seq_lat_n['Vsm'].sum(axis=2) \
                         + seq_lat_n['xsm'].dot(seq_lat_n['xsm'].T)
        y = np.hstack(seq['y'])
        xsm = np.hstack(seq_lat['xsm'])
        sum_yxtrans = y.dot(xsm.T)
        sum_xall = xsm.sum(axis=1)[:, np.newaxis]
        sum_yall = y.sum(axis=1)[:, np.newaxis]

        # term is (xDim+1) x (xDim+1)
        term = np.vstack([np.hstack([sum_p_auto, sum_xall]),
                          np.hstack([sum_xall.T, t.sum().reshape((1, 1))])])
        cd = util.rdiv(np.hstack([sum_yxtrans, sum_yall]), term)  # yDim x (xDim+1)

        params['C'] = cd[:, :x_dim]
        params['d'] = cd[:, -1]

        # yCent must be based on the new d
        # yCent = bsxfun(@minus, [seq.y], currentParams.d);
        # R = (yCent * yCent' - (yCent * [seq.xsm]') * currentParams.C')
        #     / sum(T);
        c = params['C']
        d = params['d'][:, np.newaxis]
        if params['notes']['RforceDiagonal']:
            sum_yytrans = (y * y).sum(axis=1)[:, np.newaxis]
            yd = sum_yall * d
            term = ((sum_yxtrans-d.dot(sum_xall.T))*c).sum(axis=1)[:, np.newaxis]
            r = d**2 + (sum_yytrans-2*yd-term)/t.sum()

            # Set minimum private variance
            r = np.maximum(var_floor, r)
            params['R'] = np.diag(r[:, 0])
        else:
            sum_yytrans = y.dot(y.T)
            yd = sum_yall.dot(d.T)
            term = (sum_yxtrans - d.dot(sum_xall.T)).dot(c.T)
            r = d.dot(d.T) + (sum_yytrans - yd - yd.T - term) / t.sum()

            params['R'] = (r + r.T) / 2  # ensure symmetry

        if params['notes']['learnKernelParams']:
            res = learn_gp_params(seq_lat, params, verbose=verbose)
            if params['covType'] == 'rbf':
                params['gamma'] = res['gamma']
            elif params['covType'] == 'tri':
                params['a'] = res['a']
            elif params['covType'] == 'logexp':
                params['a'] = res['a']
            if params['notes']['learnGPNoise']:
                params['eps'] = res['eps']

        t_end = time.time() - tic
        iter_time.append(t_end)

        # Display the most recent likelihood that was evaluated
        if verbose:
            if get_ll:
                print('       lik {0} ({1:.1f} sec)'.format(ll, t_end))
            else:
                print()
        else:
            if get_ll:
                print('       lik {0}\r'.format(ll))
            else:
                print()

        # Verify that likelihood is growing monotonically
        if i <= 2:
            ll_base = ll
        elif ll < ll_old:
            print('\nError: Data likelihood has decreased ',
                  'from {0} to {1}'.format(ll_old, ll))
        elif (ll - ll_base) < (1 + tol) * (ll_old - ll_base):
            break

    print()

    if len(lls) < em_max_iters:
        print('Fitting has converged after {0} EM iterations.)'.format(len(lls)))

    if np.any(np.diag(params['R']) == var_floor):
        print('Warning: Private variance floor used ',
              'for one or more observed dimensions in GPFA.')

    return params, seq_lat, lls, iter_time


def gpfa_engine(seq_train, seq_test, x_dim=8, bin_width=20.0, tau_init=100.0,
                eps_init=1.0E-3, min_var_frac=0.01, em_max_iters=500):
    """Extract neural trajectories using GPFA.

    Parameters
    ----------
    seq_train : numpy recarray
                training data structure, whose n-th element (corresponding to the n-th experimental trial) has fields
                    trialId :
                              unique trial identifier
                    T : int
                        number of bins
                    y : ndarray of shape (#units, #bins)
                        neural data
    seq_test : numpy recarray
               test data structure (same format as seqTrain)
    x_dim : int, optional
            state dimensionality (default: 3)
    bin_width : int, optional
                spike bin width in msec (default: 20)
    tau_init : float, optional
               GP timescale initialization in msec (default: 100)
    eps_init : float, optional
               GP noise variance initialization (default: 1e-3)
    min_var_frac : float, optional
                   fraction of overall data variance for each observed
                   dimension to set as the private variance floor.  This is
                   used to combat Heywood cases, where ML parameter learning
                   returns one or more zero private variances. (default: 0.01)
                   (See Martin & McDonald, Psychometrika, Dec 1975.)
    em_max_iters : int, optional
                   number of EM iterations to run (default: 500)

    Returns
    -------
    results : dict
              results of GPFA model fitting contained in fields
              estParams : dict
                          estimated GPFA model parameters
                              covType : {'rbf', 'tri', 'logexp'}
                                        type of GP covariance
                              gamma : ndarray of shape (1, #latent_vars)
                                      related to GP timescales by 'bin_width / sqrt(gamma)'
                              eps : ndarray of shape (1, #latent_vars)
                                    GP noise variances
                              d : ndarray of shape (#units, 1)
                                  observation mean
                              C : ndarray of shape (#units, #latent_vars)
                                  mapping between the neuronal data space and the latent variable space
                              R : ndarray of shape (#units, #latent_vars)
                                  observation noise covariance
              seqTrain : numpy recarray
                         a copy of the training data structure, augmented by new fields
                             xsm : ndarray of shape (#latent_vars x #bins)
                                   posterior mean of latent variables at each time bin
                             Vsm : ndarray of shape (#latent_vars, #latent_vars, #bins)
                                   posterior covariance between latent variables at each timepoint
                             VsmGP : ndarray of shape (#bins, #bins, #latent_vars)
                                     posterior covariance over time for each latent variable
              ...
    """
    # For compute efficiency, train on equal-length segments of trials
    seq_train_cut = util.cut_trials(seq_train)
    if len(seq_train_cut) == 0:
        print( 'WARNING: no segments extracted for training.',
               ' Defaulting to segLength=Inf.')
        seq_train_cut = util.cut_trials(seq_train, seg_length=np.inf)

    # ==================================
    # Initialize state model parameters
    # ==================================
    params_init = dict()
    params_init['covType'] = 'rbf'
    # GP timescale
    # Assume binWidth is the time step size.
    params_init['gamma'] = (bin_width/tau_init)**2 * np.ones(x_dim)
    # GP noise variance
    params_init['eps'] = eps_init * np.ones(x_dim)

    # ========================================
    # Initialize observation model parameters
    # ========================================
    print('Initializing parameters using factor analysis...')

    y_all = np.hstack(seq_train_cut['y'])
    fa = FactorAnalysis(n_components=x_dim, copy=True, noise_variance_init=np.diag(np.cov(y_all, bias=True)))
    fa.fit(y_all.T)
    params_init['d'] = y_all.mean(axis=1)
    params_init['C'] = fa.components_.T
    params_init['R'] = np.diag(fa.noise_variance_)

    # Define parameter constraints
    params_init['notes'] = {
        'learnKernelParams': True,
        'learnGPNoise': False,
        'RforceDiagonal': True,
    }

    # =====================
    # Fit model parameters
    # =====================
    print ('\nFitting GPFA model...')

    params_est, seq_train_cut, ll_cut, iter_time = em(params_init, seq_train_cut, min_var_frac=min_var_frac,
                                                      em_max_iters=em_max_iters)

    # Extract neural trajectories for original, unsegmented trials
    # using learned parameters
    seq_train, ll_train = exact_inference_with_ll(seq_train, params_est)

    if len(seq_test) > 0:
        pass
        # TODO: include the assessment of generalization performance
        # % ==================================
        # % Assess generalization performance
        # % ==================================
        # if ~isempty(seqTest) % check if there are any test trials
        #     % Leave-neuron-out prediction on test data
        #     if estParams.notes.RforceDiagonal
        #         seqTest = cosmoother_gpfa_viaOrth_fast(seqTest, estParams, 1:xDim);
        #     else
        #         seqTest = cosmoother_gpfa_viaOrth(seqTest, estParams, 1:xDim);
        #     end
        #     % Compute log-likelihood of test data
        #     [blah, LLtest] = exactInferenceWithLL(seqTest, estParams);
        # end

    results = {'parameter_estimates': params_est, 'iteration_time': iter_time,
               'seqTrain': seq_train, 'log_likelihood': ll_train}
    return results


def two_stage_engine(seqTrain, seqTest, typ='fa', xDim=3, binWidth=20.0):
    return {}
