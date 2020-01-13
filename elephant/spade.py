"""
SPADE is the combination of a mining technique and multiple statistical tests
to detect and assess the statistical significance of repeated occurrences of
spike sequences (spatio-temporal patterns, STP).

Given a list of Neo Spiketrain objects, assumed to be recorded in parallel, the
SPADE analysis can be applied as demonstrated in this short toy example of 10
artificial spike trains of exhibiting fully synchronous events of order 10.

This modules relies on the implementation of the fp-growth algorithm contained
in the file fim.so which can be found here (http://www.borgelt.net/pyfim.html)
and should be available in the spade_src folder (elephant/spade_src/).
If the fim.so module is not present in the correct location or cannot be
imported (only available for linux OS) SPADE will make use of a python
implementation of the fast fca algorithm contained in
elephant/spade_src/fast_fca.py, which is about 10 times slower.


import elephant.spade
import elephant.spike_train_generation
import quantities as pq

# Generate correlated data
sts = elephant.spike_train_generation.cpp(
    rate=5*pq.Hz, A=[0]+[0.99]+[0]*9+[0.01], t_stop=10*pq.s)

# Mining patterns with SPADE using a binsize of 1 ms and a window length of 1
# bin (i.e., detecting only synchronous patterns).
patterns = spade.spade(
        data=sts, binsize=1*pq.ms, winlen=1, dither=5*pq.ms,
        min_spikes=10, n_surr=10, psr_param=[0,0,3],
        output_format='patterns')['patterns'][0]

# Plotting
plt.figure()
for neu in patterns['neurons']:
    if neu == 0:
        plt.plot(
            patterns['times'], [neu]*len(patterns['times']), 'ro',
            label='pattern')
    else:
        plt.plot(
            patterns['times'], [neu] * len(patterns['times']), 'ro')
# Raster plot of the data
for st_idx, st in enumerate(sts):
    if st_idx == 0:
        plt.plot(st.rescale(pq.ms), [st_idx] * len(st), 'k.', label='spikes')
    else:
        plt.plot(st.rescale(pq.ms), [st_idx] * len(st), 'k.')
plt.ylim([-1, len(sts)])
plt.xlabel('time (ms)')
plt.ylabel('neurons ids')
plt.legend()
plt.show()

:copyright: Copyright 2017 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""
import numpy as np
from scipy import sparse
import operator
import neo
import elephant.spike_train_surrogates as surr
import elephant.conversion as conv
from itertools import chain, combinations
from functools import reduce
import time
import quantities as pq
import warnings
from elephant.spade_src import fast_fca
from collections import defaultdict

warnings.simplefilter('once', UserWarning)

try:
    from mpi4py import MPI  # for parallelized routines

    HAVE_MPI = True
except ImportError:  # pragma: no cover
    HAVE_MPI = False

try:
    from elephant.spade_src import fim
#
#    HAVE_FIM = True
#except ImportError:  # pragma: no cover
#    HAVE_FIM = False
except ImportError:  # pragma: no cover
    import fim
HAVE_FIM = True


def spade(data, binsize, winlen, min_spikes=2, min_occ=2, max_spikes=None,
          max_occ=None, min_neu=1, n_subsets=0, delta=0, epsilon=0,
          stability_thresh=None, n_surr=0, dither=15 * pq.ms, spectrum='#',
          alpha=1, stat_corr='fdr', psr_param=None, output_format='concepts'):
    """
    Perform the SPADE [1,2] analysis for the parallel spike trains given in the
    input. The data are discretized with a temporal resolution equal binsize
    in a sliding window of winlen*binsize milliseconds.

    First, spike patterns are mined from the data using a technique termed
    frequent itemset mining (FIM) or formal concept analysis (FCA). In this
    framework, a particular spatio-temporal spike pattern is termed a
    "concept". It is then possible to compute the stability and the signature
    significance of all pattern candidates. In a final step, it is possible to
    select a stability threshold and the significance level to select only
    stable/significant concepts.

    Parameters
    ----------
    data: list of neo.SpikeTrains
        List containing the parallel spike trains to analyze
    binsize: Quantity
        The time precision used to discretize the data (binning).
    winlen: int (positive)
        The size (number of bins) of the sliding window used for the analysis.
        The maximal length of a pattern (delay between first and last spike) is
        then given by winlen*binsize
    min_spikes: int (positive)
        Minimum number of spikes of a sequence to be considered a pattern.
        Default: 2
    min_occ: int (positive)
        Minimum number of occurrences of a sequence to be considered as a
        pattern.
        Default: 2
    max_spikes: int (positive)
        Maximum number of spikes of a sequence to be considered a pattern. If
        None no maximal number of spikes is considered.
        Default: None
    max_occ: int (positive)
        Maximum number of occurrences of a sequence to be considered as a
        pattern. If None, no maximal number of occurrences is considered.
        Default: None
    min_neu: int (positive)
        Minimum number of neurons in a sequence to considered a pattern.
        Default: 1
    n_subsets: int
        Number of subsets of a concept used to approximate its stability. If
        n_subset is set to 0 the stability is not computed. If, however,
        for parameters delta and epsilon (see below) delta + epsilon == 0,
        then an optimal n_subsets is calculated according to the formula given
        in Babin, Kuznetsov (2012), proposition 6:

         ..math::
                n_subset = frac{1}{2\eps^2} \ln(frac{2}{\delta}) +1

        Default:0
    delta: float
        delta: probability with at least ..math:$1-\delta$
        Default: 0
    epsilon: float
        epsilon: absolute error
        Default: 0
    stability_thresh: None or list of float
        List containing the stability thresholds used to filter the concepts.
        If stab_thr is None, then the concepts are not filtered. Otherwise,
        only concepts with intensional stability > stab_thr[0] or extensional
        stability > stab_thr[1] are returned and used for further analysis
        within SPADE.
        Default: None
    n_surr: int
        Number of surrogates to generate to compute the p-value spectrum.
        This number should be large (n_surr>=1000 is recommended for 100
        spike trains in *sts*). If n_surr is 0, then the p-value spectrum is
        not computed.
        Default: 0
    dither: Quantity
        Amount of spike time dithering for creating the surrogates for
        filtering the pattern spectrum. A spike at time t is placed randomly
        within ]t-dither, t+dither[ (see also
        elephant.spike_train_surrogates.dither_spikes).
        Default: 15*pq.ms
    spectrum: str
        Define the signature of the patterns, it can assume values:
        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrences)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)
        Default: '#'
    alpha: float
        The significance level of the hypothesis tests performed. If alpha=1
        all the concepts are returned. If 0<alpha<1 the concepts
        are filtered according to their signature in the p-value spectrum.
        Default: 1
    stat_corr: str
        Statistical correction to be applied:
            '', 'no' : no statistical correction
            'f', 'fdr' : false discovery rate
            'b', 'bonf': Bonferroni correction
            'hb', 'holm_bonf': Holm-Bonferroni correction
         Default: 'fdr'
    psr_param: None or list of int
        This list contains parameters used in the pattern spectrum filtering:
            psr_param[0]: correction parameter for subset filtering
                (see parameter h of pattern_set_reduction()).
            psr_param[1]: correction parameter for superset filtering
                (see parameter k of pattern_set_reduction()).
            psr_param[2]: correction parameter for covered-spikes criterion
                (see parameter l for pattern_set_reduction()).
    output_format: str
        distinguish the format of the output (see Returns). Can assume values
        'concepts' and 'patterns'.

    Returns
    -------
    The output depends on the value of the parameter output_format.

    If output_format is 'concepts':
        output: dict
            Dictionary containing the following keys:
            patterns: tuple
                Each element of the tuple corresponds to a pattern and is
                itself a tuple consisting of:
                    (spikes in the pattern, occurrences of the patterns)
                For details see function concepts_mining().

                If n_subsets>0:
                    (spikes in the pattern, occurrences of the patterns,
                    (intensional stability, extensional stability))
                    corresponding pvalue

            The patterns are filtered depending on the parameters in input:
            If stability_thresh==None and alpha==1:
                output['patterns'] contains all the candidates patterns
                (all concepts mined with the fca algorithm)
            If stability_thresh!=None and alpha==1:
                output contains only patterns candidates with:
                    intensional stability>stability_thresh[0] or
                    extensional stability>stability_thresh[1]
            If stability_thresh==None and alpha!=1:
                output contains only pattern candidates with a signature
                significant in respect the significance level alpha corrected
            If stability_thresh!=None and alpha!=1:
                output['patterns'] contains only pattern candidates with a
                signature significant in respect the significance level alpha
                corrected and such that:
                    intensional stability>stability_thresh[0] or
                    extensional stability>stability_thresh[1]
                In addition, output['non_sgnf_sgnt'] contains the list of
                non-significant signature for the significance level alpha.
            If n_surr>0:
                output['pvalue_spectrum'] contains a tuple of signatures and
                the corresponding p-value.

    If output_format is 'patterns':
        output: list
            List of dictionaries. Each dictionary corresponds to a patterns and
            has the following keys:
                neurons: array containing the indices of the neurons of the
                    pattern.
                lags: array containing the lags (integers corresponding to the
                    number of bins) between the spikes of the patterns. The
                    first lag is always assumed to be 0 and corresponds to the
                    first spike ['times'] array containing the times.
            (integers corresponding to the bin idx) of the occurrences of the
            patterns
                signature: tuple containing two integers:
                    (number of spikes of the patterns,
                    number of occurrences of the pattern)
            pvalue: the p-value corresponding to the pattern. If n_surr==0 the
                p-values are set to 0.0.

    Notes
    -----
    If detected, this function will utilize MPI to parallelize the analysis.

    Example
    -------
    The following applies SPADE to a list of spike trains in data. These calls
    do not include the statistical testing (for details see the documentation
    of spade.spade())

    >>> import elephant.spade
    >>> import quantities as pq
    >>> binsize = 3 * pq.ms # time resolution used to discretize the data
    >>> winlen = 10 # maximal pattern length in bins (i.e., sliding window)
    >>> result_spade = spade.spade(data, binsize, winlen)

    References
    ----------
    [1] Torre, E., Picado-Muino, D., Denker, M., Borgelt, C., & Gruen, S.(2013)
     Statistical evaluation of synchronous spike patterns extracted by
     frequent item set mining. Frontiers in Computational Neuroscience, 7.
    [2] Quaglio, P., Yegenoglu, A., Torre, E., Endres, D. M., & Gruen, S.(2017)
     Detection and Evaluation of Spatio-Temporal Spike Patterns in Massively
     Parallel Spike Train Data with SPADE.
    Frontiers in Computational Neuroscience, 11.
    """
    if HAVE_MPI:  # pragma: no cover
        comm = MPI.COMM_WORLD  # create MPI communicator
        rank = comm.Get_rank()  # get rank of current MPI task
    else:
        rank = 0
    output = {}
    # Decide whether compute pvalue spectrum
    if n_surr > 0:
        # Compute pvalue spectrum
        time_pvalue_spectrum = time.time()
        pv_spec = pvalue_spectrum(data, binsize, winlen, dither=dither,
                                  n_surr=n_surr, min_spikes=min_spikes,
                                  min_occ=min_occ, max_spikes=max_spikes,
                                  max_occ=max_occ, min_neu=min_neu,
                                  spectrum=spectrum)
        time_pvalue_spectrum = time.time() - time_pvalue_spectrum
        print("Total time for pvalue spectrum computation: {}".format(
            time_pvalue_spectrum))
        # Storing pvalue spectrum
        output['pvalue_spectrum'] = pv_spec
    elif 0 < alpha < 1:
        warnings.warn('0<alpha<1 but p-value spectrum has not been '
                      'computed (n_surr==0)')
    time_mining = time.time()
    # Decide if compute the approximated stability
    if n_subsets > 0:
        # Mine the data for extraction of concepts
        concepts, rel_matrix = concepts_mining(data, binsize, winlen,
                                               min_spikes=min_spikes,
                                               min_occ=min_occ,
                                               min_neu=min_neu,
                                               max_spikes=max_spikes,
                                               max_occ=max_occ,
                                               report='a')
        time_mining = time.time() - time_mining
        print("Time for data mining: {}".format(time_mining))
        # Computing the approximated stability of all the concepts
        time_stability = time.time()
        concepts = approximate_stability(concepts, rel_matrix, n_subsets,
                                         delta=delta, epsilon=epsilon)
        time_stability = time.time() - time_stability
        print("Time for stability computation: {}".format(time_stability))
        # Filtering the concepts using stability thresholds
        if stability_thresh is not None:
            concepts = list(filter(
                lambda c: _stability_filter(c, stability_thresh), concepts))
    elif stability_thresh is not None:
        warnings.warn('Stability_thresh not None but stability has not been '
                      'computed (n_subsets==0)')
    elif rank == 0:
        # Mine the data for extraction of concepts
        concepts, rel_matrix = concepts_mining(data, binsize, winlen,
                                               min_spikes=min_spikes,
                                               min_occ=min_occ,
                                               max_spikes=max_spikes,
                                               max_occ=max_occ,
                                               min_neu=min_neu,
                                               report='a')
        time_mining = time.time() - time_mining
        print("Time for data mining: {}".format(time_mining))
    print("Time for FIM Input Preprocessing: {}".format(
        concepts_mining.time_pre / concepts_mining.num_runs))
    print("Time for FIM: {}".format(
        concepts_mining.time_fim / concepts_mining.num_runs))
    print("Time for FIM Output Postprocessing: {}".format(
        concepts_mining.time_post / concepts_mining.num_runs))
    if rank == 0:
        # Decide whether filter concepts with psf
        if n_surr > 0:
            if len(pv_spec) == 0:
                ns_sgnt = []
            else:
                # Computing non-significant entries of the spectrum applying
                # the statistical correction
                time_significance = time.time()
                ns_sgnt = test_signature_significance(pv_spec, alpha,
                                                      corr=stat_corr,
                                                      report='non_significant',
                                                      spectrum=spectrum)
                time_significance = time.time() - time_significance
                print(
                    "Time for multiple testing: {}".format(time_significance))
            # Storing non-significant entries of the pvalue spectrum
            output['non_sgnf_sgnt'] = ns_sgnt
            # Filter concepts with pvalue spectrum (psf)
            if len(ns_sgnt) > 0:
                time_psf = time.time()
                concepts = list(filter(
                    lambda c: _pattern_spectrum_filter(
                        c, ns_sgnt, spectrum, winlen), concepts))
                time_psf = time.time() - time_psf
                print("Time for psf: {}".format(time_psf))
                # Decide whether to filter concepts using psr
                if psr_param is not None:
                    # Filter using conditional tests (psr)
                    time_psr = time.time()
                    concepts = pattern_set_reduction(concepts, ns_sgnt,
                                                     winlen=winlen,
                                                     h=psr_param[0],
                                                     k=psr_param[1],
                                                     l=psr_param[2],
                                                     min_spikes=min_spikes,
                                                     min_occ=min_occ)
                    time_psr = time.time() - time_psr
                    print("Time for psr: {}".format(time_psr))

        # Storing patterns
        if output_format == 'patterns':
            # If the p-value spectra was not computed, is set to an empty list
            if n_surr == 0:
                pv_spec = None
            # Transforming concepts to dictionary containing pattern's infos
            time_to_patterns = time.time()
            output['patterns'] = concept_output_to_patterns(concepts,
                                                            winlen, binsize,
                                                            pv_spec, spectrum,
                                                            data[0].t_start)
            time_to_patterns = time.time() - time_to_patterns
            print("Time for converting concepts to patterns: {}".format(
                time_to_patterns))
        elif output_format == 'concepts':
            output['patterns'] = concepts
        else:
            raise ValueError(
                "The output_format value has to be one between"
                "'patterns' and 'concepts'")
        return output
    # rank!=0 returning None
    else:
        warnings.warn('Returning None because executed on a process != 0')
        return None


def concepts_mining(data, binsize, winlen, min_spikes=2, min_occ=2,
                    max_spikes=None, max_occ=None, min_neu=1, report='a'):
    """
    Find pattern candidates extracting all the concepts of the context formed
    by the objects defined as all windows of length winlen*binsize slided
    along the data and the attributes as the spikes occurring in each of the
    window discretized at a time resolution equal to binsize. Hence, the output
    are all the repeated sequences of spikes with maximal length winlen, which
    are not trivially explained by the same number of occurrences of a superset
    of spikes.

    Parameters
    ----------
    data: list of neo.SpikeTrains
        List containing the parallel spike trains to analyze
    binsize: Quantity
        The time precision used to discretize the data (binning).
    winlen: int (positive)
        The size (number of bins) of the sliding window used for the analysis.
        The maximal length of a pattern (delay between first and last spike) is
        then given by winlen*binsize
    min_spikes: int (positive)
        Minimum number of spikes of a sequence to be considered a pattern.
        Default: 2
    min_occ: int (positive)
        Minimum number of occurrences of a sequence to be considered as a
        pattern.
       Default: 2
    max_spikes: int (positive)
        Maximum number of spikes of a sequence to be considered a pattern. If
        None no maximal number of spikes is considered.
        Default: None
    max_occ: int (positive)
        Maximum number of occurrences of a sequence to be considered as a
        pattern. If None, no maximal number of occurrences is considered.
        Default: None
    min_neu: int (positive)
        Minimum number of neurons in a sequence to considered a pattern.
        Default: 1
    report: str
        Indicates the output of the function.
        'a': all the mined patterns
        '#': pattern spectrum using as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using as signature the triplets:
            (number of spikes, number of occurrence, difference between the
            times of the last and the first spike of the pattern)
        Default: 'a'

    Returns
    -------
    mining_results: list
        If report == 'a':
            All the pattern candidates (concepts) found in the data. Each
            pattern is represented as a tuple containing
            (spike IDs, discrete times (window position)
            of the  occurrences of the pattern). The spike IDs are defined as:
            spike_id=neuron_id*bin_id; with neuron_id in [0, len(data)] and
            bin_id in [0, winlen].
        If report == '#':
             The pattern spectrum is represented as a list of triplets each
             formed by:
                (pattern size, number of occurrences, number of patterns)
        If report == '3d#':
             The pattern spectrum is represented as a list of quadruplets each
             formed by:
                (pattern size, number of occurrences, difference between last
                and first spike of the pattern, number of patterns)
    rel_matrix : sparse.coo_matrix
        A binary matrix with shape (number of windows, winlen*len(data)). Each
        row corresponds to a window (order according to their position in
        time). Each column corresponds to one bin and one neuron and it is 0 if
        no spikes or 1 if one or more spikes occurred in that bin for that
        particular neuron. For example, the entry [0,0] of this matrix
        corresponds to the first bin of the first window position for the first
        neuron, the entry [0,winlen] to the first bin of the first window
        position for the second neuron.
    """
    # Check that data is a list of SpikeTrains
    if not all([isinstance(elem, neo.SpikeTrain) for elem in data]):
        raise TypeError(
            'data must be a list of SpikeTrains')
    # Check that all spiketrains have same t_start and same t_stop
    if not all([st.t_start == data[0].t_start for st in data]) or not all(
            [st.t_stop == data[0].t_stop for st in data]):
        raise AttributeError(
            'All spiketrains must have the same t_start and t_stop')
    if report not in ['a', '#', '3d#']:
        raise ValueError(
            "report has to assume of the following values:" +
            "  'a', '#' and '3d#,' got {} instead".format(report))
    # Binning the data and clipping (binary matrix)
    concepts_mining.num_runs += 1
    time_pre = time.time()
    binary_matrix = conv.BinnedSpikeTrain(
        data, binsize).to_sparse_bool_array().tocoo()
    # Computing the context and the binary matrix encoding the relation between
    # objects (window positions) and attributes (spikes,
    # indexed with a number equal to  neuron idx*winlen+bin idx)
    context, transactions, rel_matrix = _build_context(binary_matrix, winlen)
    # By default, set the maximum pattern size to the maximum number of
    # spikes in a window
    if max_spikes is None:
        max_spikes = len(data) * winlen
    #    max_spikes = np.max((int(np.max(np.sum(rel_matrix, axis=1))),
    #                         min_spikes + 1))
    # By default, set maximum number of occurrences to number of non-empty
    # windows
    if max_occ is None:
        max_occ = int(np.sum(np.sum(rel_matrix, axis=1) > 0))
    concepts_mining.time_pre += time.time() - time_pre
    # Check if fim.so available and use it
    if HAVE_FIM:
        # Return the output
        mining_results = _fpgrowth(
            transactions,
            rel_matrix=rel_matrix,
            min_c=min_occ,
            min_z=min_spikes,
            max_z=max_spikes,
            max_c=max_occ,
            winlen=winlen,
            min_neu=min_neu,
            report=report)
        return mining_results, rel_matrix
    # Otherwise use fast_fca python implementation
    warnings.warn(
        'Optimized C implementation of FCA (fim.so/fim.pyd) not found ' +
        'in elephant/spade_src folder, or not compatible with this ' +
        'Python version. You are using the pure Python implementation ' +
        'of fast fca.')
    # Return output
    mining_results = _fast_fca(
        context,
        min_c=min_occ,
        min_z=min_spikes,
        max_z=max_spikes,
        max_c=max_occ,
        winlen=winlen,
        min_neu=min_neu,
        report=report)
    return mining_results, rel_matrix


concepts_mining.time_pre = 0
concepts_mining.time_fim = 0
concepts_mining.time_post = 0
concepts_mining.num_runs = 0


def _build_context(binary_matrix, winlen, only_windows_with_first_spike=True):
    """
    Building the context given a matrix (number of trains x number of bins) of
    binned spike trains
    Parameters
    ----------
    binary_matrix : sparse.coo_matrix
        Binary matrix containing the binned spike trains
    winlen : int
        Length of the binsize used to bin the data
    only_windows_with_first_spike : bool
        Whether to consider every window or only the one with a spike in the
        first bin. It is possible to discard windows without a spike in the
        first bin because the same configuration of spikes will be repeated
        in a following window, just with different position for the first spike
        Default: True

    Returns
    --------
    context : list
        List of tuples containing one object (window position idx) and one of
        the correspondent spikes idx (bin idx * neuron idx)
    transactions : list
        List of all transactions, each element of the list contains the
        attributes of the corresponding object.
    rel_matrix : sparse.coo_matrix
        A binary matrix with shape (number of windows, winlen*len(data)). Each
        row corresponds to a window (order according to
        their position in time).
        Each column corresponds to one bin and one neuron and it is 0 if no
        spikes or 1 if one or more spikes occurred in that bin for that
        particular neuron.
        E.g. the entry [0,0] of this matrix corresponds to the first bin of the
        first window position for the first neuron, the entry [0,winlen] to the
        first bin of the first window position for the second neuron.
    """
    # Initialization of the outputs
    context = []
    transactions = []
    num_neurons, num_bins = binary_matrix.shape
    indices = np.argsort(binary_matrix.col)
    binary_matrix.row = binary_matrix.row[indices]
    binary_matrix.col = binary_matrix.col[indices]
    if only_windows_with_first_spike:
        # out of all window positions
        # get all non-empty first bins
        window_indices = np.unique(binary_matrix.col)
    else:
        window_indices = np.arange(num_bins - winlen + 1)
    windows_row = []
    windows_col = []
    for window_idx in window_indices:
        for col in range(window_idx, window_idx + winlen):
            if col in binary_matrix.col:
                nonzero_indices = np.nonzero(binary_matrix.col == col)[0]
                windows_col.extend(
                    binary_matrix.row[nonzero_indices] * winlen
                    + (col - window_idx))
                windows_row.extend([window_idx] * len(nonzero_indices))
    # Shape of the rel_matrix:
    # (num of window positions,
    # num of bins in one window * number of neurons)
    num_windows = window_indices.shape[0]
    rel_matrix = sparse.coo_matrix(
        (np.ones((len(windows_col)), dtype=bool),
         (windows_row, windows_col)),
        shape=(num_bins, winlen * num_neurons),
        dtype=bool).A
    # Array containing all the possible attributes (each spike is indexed by
    # a number equal to neu idx*winlen + bin_idx)
    attributes = np.array(
        [s * winlen + t for s in range(binary_matrix.shape[0])
         for t in range(winlen)])
    # Building context and rel_matrix
    # Looping all the window positions w
    for w in window_indices:
        # spikes in the current window
        times = rel_matrix[w]
        current_transactions = attributes[times]
        # adding to the context the window positions and the correspondent
        # attributes (spike idx) (fast_fca input)
        context += [(w, a) for a in current_transactions]
        # appending to the transactions spike idx (fast_fca input) of the
        # current window (fpgrowth input)
        transactions.append(list(current_transactions))
    # Return context and rel_matrix
    return context, transactions, rel_matrix


def _fpgrowth(transactions, min_c=2, min_z=2, max_z=None,
              max_c=None, rel_matrix=None, winlen=1, min_neu=1,
              target='c', report='a'):
    """
    Find frequent item sets with the fpgrowth algorithm.

    Parameters
    ----------
    transactions: tuple
                Transactions database to mine.
                The database must be an iterable of transactions;
                each transaction must be an iterable of items;
                each item must be a hashable object.
                If the database is a dictionary, the transactions are
                the keys, the values their (integer) multiplicities.
    target: str
            type of frequent item sets to find
            s/a   sets/all   all     frequent item sets
            c     closed     closed  frequent item sets
            m     maximal    maximal frequent item sets
            g     gens       generators
            Default:'c'
    min_c: int
        minimum support of an item set
        Default: 2
    min_z: int
         minimum number of items per item set
        Default: 2
    max_z: None/int
         maximum number of items per item set. If max_c==None no maximal
         size required
        Default: None
    max_c: None/int
         maximum support per item set. If max_c==None no maximal
         support required
        Default: None
    report: str
        'a': all the mined patterns
        '#': pattern spectrum using as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using as signature the triplets:
            (number of spikes, number of occurrence, difference between the
            times of the last and the first spike of the pattern)
        Default: 'a'
    rel_matrix : None or sparse.coo_matrix
        A binary matrix with shape (number of windows, winlen*len(data)). Each
        row corresponds to a window (order according to
        their position in time).
        Each column corresponds to one bin and one neuron and it is 0 if no
        spikes or 1 if one or more spikes occurred in that bin for that
        particular neuron.
        E.g. the entry [0,0] of this matrix corresponds to the first bin of the
        first window position for the first neuron, the entry [0,winlen] to the
        first bin of the first window position for the second neuron.
        If == None only the closed frequent itemsets (intent) are returned and
        not which the index of their occurrences (extent)
        Default: None
    The following parameters are specific to Massive parallel SpikeTrains
    winlen: int (positive)
        The size (number of bins) of the sliding window used for the
        analysis. The maximal length of a pattern (delay between first and
        last spike) is then given by winlen*binsize
        Default: 1
    min_neu: int (positive)
         Minimum number of neurons in a sequence to considered a
         potential pattern.
         Default: 1

    Returns
    --------
    If report == 'a':
        All the pattern candidates (concepts) found in the data. Each
        pattern is represented as a tuple containing
        (spike IDs, discrete times (window position)
        of the  occurrences of the pattern). The spike IDs are defined as:
        spike_id=neuron_id*bin_id; with neuron_id in [0, len(data)] and
        bin_id in [0, winlen].
    If report == '#':
         The pattern spectrum is represented as a list of triplets each
         formed by:
            (pattern size, number of occurrences, number of patterns)
    If report == '3d#':
         The pattern spectrum is represented as a list of quadruplets each
         formed by:
            (pattern size, number of occurrences, difference between last
            and first spike of the pattern, number of patterns)

    """
    # By default, set the maximum pattern size to the number of spiketrains
    if max_z is None:
        max_z = np.max((np.max([len(tr) for tr in transactions]), min_z + 1))
    # By default set maximum number of data to number of bins
    if max_c is None:
        max_c = len(transactions)
    if min_neu >= 1:
        # Initializing outputs
        concepts = []
        if report == '#':
            spec_matrix = np.zeros((max_z + 1, max_c + 1))
        if report == '3d#':
            spec_matrix = np.zeros((max_z + 1, max_c + 1, winlen))
        spectrum = []
        # Mining the data with fpgrowth algorithm
        time_fim = time.time()
        if np.unique(transactions, return_counts=True)[1][0] == len(
                transactions):
            fpgrowth_output = [(tuple(transactions[0]), len(transactions))]
        else:
            fpgrowth_output = fim.fpgrowth(
                tracts=transactions,
                target=target,
                supp=-min_c,
                zmin=min_z,
                zmax=max_z,
                report='a',
                algo='s')
        concepts_mining.time_fim += time.time() - time_fim
        time_post = time.time()
        # Applying min/max conditions and computing extent (window positions)
        fpgrowth_output = list(filter(
            lambda c: _fpgrowth_filter(
                c, winlen, max_c, min_neu), fpgrowth_output))
        # filter out subsets of patterns that are found as a side-effect
        # of using the moving window strategy
        fpgrowth_output = _filter_for_moving_window_subsets(
            fpgrowth_output, winlen)
        for (intent, supp) in fpgrowth_output:
            if report == 'a':
                if rel_matrix is not None:
                    # Computing the extent of the concept (patterns
                    # occurrences), checking in rel_matrix in which windows
                    # the intent occurred
                    extent = tuple(np.where(
                        np.all(rel_matrix[:, intent], axis=1) == 1)[0])
                concepts.append((intent, extent))
            # Computing 2d spectrum
            elif report == '#':
                spec_matrix[len(intent) - 1, supp - 1] += 1
            # Computing 3d spectrum
            elif report == '3d#':
                spec_matrix[len(intent) - 1, supp - 1, max(
                    np.array(intent) % winlen)] += 1
        del fpgrowth_output
        if report == 'a':
            concepts_mining.time_post += time.time() - time_post
            return concepts
        # TODO: Do we really need the number of patterns here
        #  or is it a unused quantity?
        #  I suppose, it was used in an older version of the p-value spectrum.
        else:
            if report == '#':
                for (z, c) in np.transpose(np.where(spec_matrix != 0)):
                    spectrum.append((z + 1, c + 1, int(spec_matrix[z, c])))

            elif report == '3d#':
                for (z, c, l) in np.transpose(np.where(spec_matrix != 0)):
                    spectrum.append(
                        (z + 1, c + 1, l, int(spec_matrix[z, c, l])))
            del spec_matrix

            if len(spectrum) > 0:
                spectrum = np.array(spectrum)
            elif report == '#':
                spectrum = np.zeros(shape=(0, 3))
            elif report == '3d#':
                spectrum = np.zeros(shape=(0, 4))

            concepts_mining.time_post += time.time() - time_post
            return spectrum
    else:
        raise AttributeError('min_neu must be an integer >=1')


def _fpgrowth_filter(concept, winlen, max_c, min_neu):
    """
    Filter for selecting closed frequent items set with a minimum number of
    neurons and a maximum number of occurrences and first spike in the first
    bin position
    """
    keep_concepts = len(
        np.unique(np.array(
            concept[0]) // winlen)) >= min_neu and concept[1] <= max_c and min(
        np.array(concept[0]) % winlen) == 0
    return keep_concepts


def _rereference_to_last_spike(transactions, winlen):
    """
    Converts transactions from the default format
    neu_idx * winlen + bin_idx (relative to window start)
    into the format
    neu_idx * winlen + bin_idx (relative to last spike)
    """
    len_transactions = len(transactions)
    neurons = np.zeros(len_transactions, dtype=int)
    bins = np.zeros(len_transactions, dtype=int)

    # extract neuron and bin indices
    for idx, attribute in enumerate(transactions):
        neurons[idx] = attribute // winlen
        bins[idx] = attribute % winlen

    # rereference bins to last spike
    bins = bins.max() - bins

    # calculate converted transactions
    converted_transactions = neurons * winlen + bins

    return converted_transactions


def _filter_for_moving_window_subsets(concepts, winlen):
    """
    Since we're using a moving window subpatterns starting from
    subsequent spikes after the first pattern spike will also be found.
    This filter removes them if they do not occur on their own in
    addition to the occurrences explained by their superset.
    Uses a reverse map with a set representation.
    """
    # don't do anything if the input list is empty
    if not len(concepts):
        return concepts

    if hasattr(concepts[0], 'intent'):
        # fca format
        # sort the concepts by (decreasing) support
        concepts.sort(key=lambda c: -len(c.extent))

        support = np.array([len(c.extent) for c in concepts])

        # convert transactions relative to last pattern spike
        converted_transactions = [_rereference_to_last_spike(c.intent,
                                                             winlen=winlen)
                                  for c in concepts]
    else:
        # fim.fpgrowth format
        # sort the concepts by (decreasing) support
        concepts.sort(key=lambda c: -c[1])

        support = np.array([c[1] for c in concepts])

        # convert transactions relative to last pattern spike
        converted_transactions = [_rereference_to_last_spike(c[0],
                                                             winlen=winlen)
                                  for c in concepts]

    output = []

    for current_support in np.unique(support):
        support_indices = np.nonzero(support == current_support)[0]

        # construct reverse map
        reverse_map = defaultdict(set)
        for map_idx, i in enumerate(support_indices):
            for window_bin in converted_transactions[i]:
                reverse_map[window_bin].add(map_idx)

        for i in support_indices:
            intersection = reduce(
                operator.and_,
                (reverse_map[window_bin]
                 for window_bin in converted_transactions[i]))
            if len(intersection) == 1:
                output.append(concepts[i])

    return output


def _fast_fca(context, min_c=2, min_z=2, max_z=None,
              max_c=None, report='a', winlen=1, min_neu=1):
    """
    Find concepts of the context with the fast-fca algorithm.

    Parameters
    ----------
    context : list
        List of tuples containing one object and one the correspondent
        attribute
    min_c: int
        minimum support of an item set
        Default: 2
    min_z: int
         minimum number of items per item set
        Default: 2
    max_z: None/int
         maximum number of items per item set. If max_c==None no maximal
         size required
        Default: None
    max_c: None/int
         maximum support per item set. If max_c==None no maximal
         support required
        Default: None
    report: str
        'a': all the mined patterns
        '#': pattern spectrum using as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using as signature the triplets:
            (number of spikes, number of occurrence, difference between the
            times of the last and the first spike of the pattern)
        Default: 'a'
    The following parameters are specific to Massive parallel SpikeTrains
    winlen: int (positive)
        The size (number of bins) of the sliding window used for the
        analysis. The maximal length of a pattern (delay between first and
        last spike) is then given by winlen*binsize
        Default: 1
    min_neu: int (positive)
         Minimum number of neurons in a sequence to considered a
         potential pattern.
         Default: 1

    Returns
    --------
    If report == 'a':
        All the pattern candidates (concepts) found in the data. Each
        pattern is represented as a tuple containing
        (spike IDs, discrete times (window position)
        of the  occurrences of the pattern). The spike IDs are defined as:
        spike_id=neuron_id*bin_id; with neuron_id in [0, len(data)] and
        bin_id in [0, winlen].
    If report == '#':
         The pattern spectrum is represented as a list of triplets each
         formed by:
            (pattern size, number of occurrences, number of patterns)
    If report == '3d#':
         The pattern spectrum is represented as a list of quadruplets each
         formed by:
            (pattern size, number of occurrences, difference between last
            and first spike of the pattern, number of patterns)
    """
    # Initializing outputs
    concepts = []
    # Check parameters
    if min_neu < 1:
        raise AttributeError('min_neu must be an integer >=1')
    # By default set maximum number of attributes
    if max_z is None:
        max_z = len(context)
    # By default set maximum number of data to number of bins
    if max_c is None:
        max_c = len(context)
    if report == '#':
        spec_matrix = np.zeros((max_z, max_c))
    if report == '3d#':
        spec_matrix = np.zeros((max_z, max_c, winlen))
    spectrum = []
    # Mining the data with fast fca algorithm
    fca_out = fast_fca.FormalConcepts(context)
    fca_out.computeLattice()
    fca_concepts = fca_out.concepts
    fca_concepts = list(filter(
        lambda c: _fca_filter(
            c, winlen, min_c, min_z, max_c, max_z, min_neu), fca_concepts))
    fca_concepts = _filter_for_moving_window_subsets(fca_concepts, winlen)
    # Applying min/max conditions
    for fca_concept in fca_concepts:
        intent = tuple(fca_concept.intent)
        extent = tuple(fca_concept.extent)
        concepts.append((intent, extent))
        # computing spectrum
        if report == '#':
            spec_matrix[len(intent) - 1, len(extent) - 1] += 1
        if report == '3d#':
            spec_matrix[len(intent) - 1, len(extent) - 1, max(
                np.array(intent) % winlen)] += 1
    if report == 'a':
        return concepts
    else:
        del concepts
        # returning spectrum
        if report == '#':
            for (z, c) in np.transpose(np.where(spec_matrix != 0)):
                spectrum.append((z + 1, c + 1, int(spec_matrix[z, c])))

        if report == '3d#':
            for (z, c, l) in np.transpose(np.where(spec_matrix != 0)):
                spectrum.append(
                    (z + 1, c + 1, l, int(spec_matrix[z, c, l])))
        del spec_matrix
        return spectrum


def _fca_filter(concept, winlen, min_c, min_z, max_c, max_z, min_neu):
    """
    Filter to select concepts with minimum/maximum number of spikes and
    occurrences and first spike in the first bin position
    """
    intent = tuple(concept.intent)
    extent = tuple(concept.extent)
    keep_concepts = len(intent) >= min_z and len(extent) >= min_c and len(
        intent) <= max_z and len(extent) <= max_c and len(
        np.unique(np.array(intent) // winlen)) >= min_neu and min(
        np.array(intent) % winlen) == 0
    return keep_concepts


def pvalue_spectrum(data, binsize, winlen, dither, n_surr, min_spikes=2,
                          min_occ=2, max_spikes=None, max_occ=None, min_neu=1,
                          spectrum='#'):
    """
    Compute the p-value spectrum of pattern signatures extracted from
    surrogates of parallel spike trains, under the null hypothesis of
    independent spiking.

    * n_surr surrogates are obtained from each spike train by spike dithering
    * pattern candidates (concepts) are collected from each surrogate data
    * the signatures (number of spikes, number of occurrences) of all patterns
      are computed, and their  occurrence probability estimated by their
      occurrence frequency (p-value spectrum)


    Parameters
    ----------
    data: list of neo.SpikeTrains
        List containing the parallel spike trains to analyze
    binsize: Quantity
        The time precision used to discretize the data (binning).
    winlen: int (positive)
        The size (number of bins) of the sliding window used for the analysis.
        The maximal length of a pattern (delay between first and last spike) is
        then given by winlen*binsize
    dither: Quantity
        Amount of spike time dithering for creating the surrogates for
        filtering the pattern spectrum. A spike at time t is placed randomly
        within ]t-dither, t+dither[ (see also
        elephant.spike_train_surrogates.dither_spikes).
        Default: 15*pq.s
    n_surr: int
        Number of surrogates to generate to compute the p-value spectrum.
        This number should be large (n_surr>=1000 is recommended for 100
        spike trains in *sts*). If n_surr is 0, then the p-value spectrum is
        not computed.
        Default: 0
    min_spikes: int (positive)
        Minimum number of spikes of a sequence to be considered a pattern.
        Default: 2
    min_occ: int (positive)
       Minimum number of occurrences of a sequence to be considered as a
       pattern.
       Default: 2
    max_spikes: int (positive)
        Maximum number of spikes of a sequence to be considered a pattern. If
        None no maximal number of spikes is considered.
        Default: None
    max_occ: int (positive)
        Maximum number of occurrences of a sequence to be considered as a
        pattern. If None, no maximal number of occurrences is considered.
        Default: None
    min_neu: int (positive)
        Minimum number of neurons in a sequence to considered a pattern.
        Default: 1
    spectrum: str
        Defines the signature of the patterns, it can assume values:
        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)
        Default: '#'

    Returns
    ------
    pv_spec: list
        if spectrum == '#':
            A list of triplets (z,c,p), where (z,c) is a pattern signature
            and p is the corresponding p-value (fraction of surrogates
            containing signatures (z*,c*)>=(z,c)).
        if spectrum == '3d#':
            A list of triplets (z,c,l,p), where (z,c,l) is a pattern signature
            and p is the corresponding p-value (fraction of surrogates
            containing signatures (z*,c*,l*)>=(z,c,l)).
        Signatures whose empirical p-value is 0 are not listed.

    """
    # Initializing variables for parallel computing
    if HAVE_MPI:  # pragma: no cover
        comm = MPI.COMM_WORLD  # create MPI communicator
        rank = comm.Get_rank()  # get rank of current MPI task
        size = comm.Get_size()  # get tot number of MPI tasks
    else:
        rank = 0
        size = 1
    # Check on number of surrogates
    if n_surr <= 0:
        raise AttributeError('n_surr has to be >0')
    len_partition = n_surr // size  # length of each MPI task
    len_remainder = n_surr % size

    add_remainder = rank < len_remainder

    time_surr_generation = 0
    time_surr_mining = 0
    time_surr_spectrum = 0

    if max_spikes is None:
        # if max_spikes not defined, set it to the number of spiketrains times
        # number of bins per window.
        max_spikes = len(data) * winlen

    if spectrum == '#':
        max_occs = np.empty(shape=(len_partition + add_remainder,
                                   max_spikes - min_spikes + 1),
                            dtype=np.uint16)
    else:
        max_occs = np.empty(shape=(len_partition + add_remainder,
                                   max_spikes - min_spikes + 1, winlen),
                            dtype=np.uint16)

    for i in range(len_partition + add_remainder):
        current_time_surr_generation = time.time()
        surrs = [surr.dither_spikes(
            xx, dither=dither, n=1)[0] for xx in data]
        time_surr_generation += time.time() - current_time_surr_generation
        # Find all pattern signatures in the current surrogate data set
        current_time_surr_mining = time.time()
        surr_concepts = concepts_mining(
            surrs, binsize, winlen, min_spikes=min_spikes,
            max_spikes=max_spikes, min_occ=min_occ, max_occ=max_occ,
            min_neu=min_neu, report=spectrum)[0]
        # The last entry of the signature is the number of times the
        # signature appeared. This entry is not needed here.
        surr_concepts = surr_concepts[:, :-1]
        time_surr_mining += time.time() - current_time_surr_mining
        current_time_surr_spectrum = time.time()

        max_occs[i] = _get_max_occ(surr_concepts, min_spikes, max_spikes,
                                   winlen, spectrum)

        time_surr_spectrum += time.time() - current_time_surr_spectrum
    print("Time for surrogate generation on rank {}: {}".format(
        rank,
        time_surr_generation / (len_partition + add_remainder)))
    print("Time for surrogate mining on rank {}: {}".format(
        rank,
        time_surr_mining / (len_partition + add_remainder)))
    print("Time for surrogate spectrum on rank {}: {}".format(
        rank,
        time_surr_spectrum / (len_partition + add_remainder)))
    # Collecting results on the first PCU

    if size != 1:
        time_mpi = time.time()

        max_occs = comm.gather(max_occs, root=0)

        if rank != 0:  # pragma: no cover
            return []

        # The gather operator gives a list out. This is rearranged as a 2 resp.
        # 3 dimensional numpy-array.
        max_occs = np.vstack(max_occs)

        time_mpi = time.time() - time_mpi
        print("Time for MPI gather: {}".format(
            time_mpi))

    time_pvalues = time.time()
    # Compute the p-value spectrum, and return it

    # TODO: When we don't need the prints anymore the result of p-value spec,
    #  can be directly returned.
    pv_spec = _get_pvalue_spec(max_occs, min_spikes, max_spikes, min_occ,
                               n_surr, winlen, spectrum)

    time_pvalues = time.time() - time_pvalues
    print("Time for last loop in pvalue_spectrum: {}".format(
        time_pvalues))
    return pv_spec


def _get_pvalue_spec(max_occs, min_spikes, max_spikes, min_occ, n_surr, winlen,
                     spectrum):
    """
    This function converts the list of maximal occurrences into the
    corresponding p-value spectrum.

    Parameters
    ----------
    max_occs: np.ndarray
    min_spikes: int
    max_spikes: int
    min_occ: int
    n_surr: int
    winlen: int
    spectrum: str
        can be '#' or '3d#'.

    Returns
    -------
    if spectrum == '#':
    List[List]:
        each entry has the form: [pattern_size, pattern_occ, p_value]
    if spectrum == '3d#':
    List[List]:
        each entry has the form:
        [pattern_size, pattern_occ, pattern_dur, p_value]
    """
    pv_spec = []
    if spectrum == '#':
        for size_id, pt_size in enumerate(range(min_spikes, max_spikes + 1)):
            max_occs_size = max_occs[:, size_id]
            counts, occs = np.histogram(
                max_occs_size,
                bins=np.arange(min_occ, np.max(max_occs_size) + 2))
            occs = occs[:-1].astype(np.uint16)

            pvalues = np.cumsum(counts[::-1])[::-1] / n_surr

            for occ_id, occ in enumerate(occs):
                pv_spec.append([pt_size, occ, pvalues[occ_id]])

    elif spectrum == '3d#':
        for size_id, pt_size in enumerate(range(min_spikes, max_spikes + 1)):
            for dur in range(winlen):
                max_occs_size_dur = max_occs[:, size_id, dur]
                counts, occs = np.histogram(
                    max_occs_size_dur,
                    bins=np.arange(min_occ,
                                   np.max(max_occs_size_dur) + 2))
                occs = occs[:-1].astype(np.uint16)

                pvalues = np.cumsum(counts[::-1])[::-1] / n_surr

                for occ_id, occ in enumerate(occs):
                    pv_spec.append([pt_size, occ, dur, pvalues[occ_id]])

    # TODO: Think of returning here a numpy array.
    #  pv_spec = np.array(pv_spec)
    return pv_spec


def _get_max_occ(surr_concepts, min_spikes, max_spikes, winlen, spectrum):
    """
    This function takes from a list of surrogate_concepts those concepts which
    have the highest occurrence for a given pattern size and duration.

    Parameters
    ----------
    surr_concepts: List[List]
    min_spikes: int
    max_spikes: int
    winlen: int

    Returns
    -------
    np.ndarray:
        Two-dimensional array. Each element corresponds to a highest occurrence
        for a specific pattern size (which range from min_spikes to max_spikes)
        and pattern duration (which range from 0 to winlen-1).
        The first axis corresponds to the pattern size the second to the
        duration.
    """

    if spectrum == '#':
        max_occ = np.zeros(shape=(max_spikes - min_spikes + 1))

        for size_id, pt_size in enumerate(range(min_spikes, max_spikes + 1)):
            concepts_for_size = surr_concepts[
                                    surr_concepts[:, 0] == pt_size][:, 1]
            max_occ[size_id] = np.max(concepts_for_size, initial=0)

        for pt_size in range(max_spikes - 1, min_spikes - 1, -1):
            size_id = pt_size - min_spikes
            max_occ[size_id] = np.max(max_occ[size_id:size_id + 2])

    elif spectrum == '3d#':
        max_occ = np.zeros(shape=(max_spikes - min_spikes + 1, winlen))

        for size_id, pt_size in enumerate(range(min_spikes, max_spikes + 1)):
            concepts_for_size = surr_concepts[
                                    surr_concepts[:, 0] == pt_size][:, 1:]

            for dur in range(winlen):
                occs = concepts_for_size[concepts_for_size[:, 1] == dur][:, 0]
                max_occ[size_id, dur] = np.max(occs, initial=0)

        for pt_size in range(max_spikes - 1, min_spikes - 1, -1):
            size_id = pt_size - min_spikes
            max_occ[size_id] = np.max(max_occ[size_id:size_id + 2], axis=0)

    return max_occ


def _stability_filter(c, stab_thr):
    """Criteria by which to filter concepts from the lattice"""
    # stabilities larger then min_st
    keep_concept = c[2] > stab_thr[0] or c[3] > stab_thr[1]
    return keep_concept


def _fdr(pvalues, alpha):
    """
    performs False Discovery Rate (FDR) statistical correction on a list of
    p-values, and assesses accordingly which of the associated statistical
    tests is significant at the desired level *alpha*

    Parameters
    ----------
    pvalues: numpy.ndarray
        array of p-values, each corresponding to a statistical test
    alpha: float
        significance level (desired FDR-ratio)

    Returns
    ------
    Returns a triplet containing:
    * an array of bool, indicating for each p-value whether it was
      significantly low or not
    * the largest p-value that was below the FDR linear threshold
      (effective confidence level). That and each lower p-value are
      considered significant.
    * the rank of the largest significant p-value

    """

    # Sort the p-values from largest to smallest
    pvs_sorted = np.sort(pvalues)[::-1]  # Sort PVs in decreasing order

    # Perform FDR on the sorted p-values
    m = len(pvalues)

    for i, pv in enumerate(pvs_sorted):  # For each PV, from the largest on
        k = m - i
        if pv <= alpha * (k * 1. / m):  # continue if PV > fdr-threshold
            break  # otherwise stop
    # this applies, when loop is not stopped due to significant pvalue
    else:
        k = 0
    thresh = alpha * (k * 1. / m)

    # Return outcome of the test, critical p-value and its order
    return pvalues <= thresh, thresh, k


def _holm_bonferroni(pvalues, alpha):
    """
    performs Holm Bonferroni statistical correction on a list of
    p-values, and assesses accordingly which of the associated statistical
    tests is significant at the desired level *alpha*

    Parameters
    ----------
    pvalues: list
       list of p-values, each corresponding to a statistical test
    alpha: float
       significance level

    Returns
    -------
    tests : list
        A list of boolean values, indicating for each p-value whether it was
        significantly low or not
   """
    id_sorted = np.argsort(pvalues)
    tests = [pval <= alpha / float(
        len(pvalues) - id_sorted[pval_idx]) for pval_idx, pval in enumerate(
        pvalues)]
    return tests


def test_signature_significance(pvalue_spectrum, alpha, corr='',
                                report='spectrum', spectrum='#'):
    """
    Compute the significance spectrum of a pattern spectrum.

    Given pvalue_spectrum as a list of triplets (z,c,p), where z is pattern
    size, c is pattern support and p is the p-value of the signature (z,c),
    this routine assesses the significance of (z,c) using the confidence level
    alpha.

    Bonferroni or FDR statistical corrections can be applied.

    Parameters
    ----------
    pvalue_spectrum: list
        A list of triplets (z,c,p), where z is pattern size, c is pattern
        support and p is the p-value of signature (z,c)
    alpha: float
        Significance level of the statistical test
    corr: str
        Statistical correction to be applied:
            '', 'no' : no statistical correction
            'f', 'fdr' : false discovery rate
            'b', 'bonf': Bonferroni correction
            'hb', 'holm_bonf': Holm-Bonferroni correction
         Default: ''
    report: str
        Format to be returned for the significance spectrum:
        'spectrum': list of triplets (z,c,b), where b is a boolean specifying
             whether signature (z,c) is significant (True) or not (False)
        'significant': list containing only the significant signatures (z,c) of
            pvalue_spectrum
        'non_significant': list containing only the non-significant signatures
        Default: '#'
    spectrum: str
        Defines the signature of the patterns, it can assume values:
        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrence)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)
        Default: '#'

    Returns
    ------
    sig_spectrum: list
        Significant signatures of pvalue_spectrum, in the format specified
        by report
    """
    # If alpha == 1 all signatures are significant
    if alpha == 1:
        return []
    x_array = np.array(pvalue_spectrum)
    # Compute significance...
    if corr in ['', 'no']:  # ...without statistical correction
        tests = x_array[:, -1] <= alpha
    elif corr in ['b', 'bonf']:  # or with Bonferroni correction
        tests = x_array[:, -1] <= alpha * 1. / len(pvalue_spectrum)
    elif corr in ['f', 'fdr']:  # or with FDR correction
        tests = _fdr(x_array[:, -1], alpha=alpha)[0]
    elif corr in ['hb', 'holm_bonf']:
        tests = _holm_bonferroni(x_array[:, -1], alpha=alpha)
    else:
        raise AttributeError(
            "Parameter corr must be either ''('no'), 'b'('bonf'), 'f'('fdr')" +
            " or 'hb'('holm_bonf'), but not '" + str(corr) + "'.")
    # Return the specified results:
    if spectrum == '#':
        if report == 'spectrum':
            return [(size, supp, test)
                    for (size, supp, pv), test in zip(pvalue_spectrum, tests)]
        elif report == 'significant':
            return [(size, supp) for ((size, supp, pv), test)
                    in zip(pvalue_spectrum, tests) if test]
        elif report == 'non_significant':
            return [
                (size, supp) for ((size, supp, pv), test) in zip(
                    pvalue_spectrum, tests) if not test]
        raise AttributeError("report must be either 'spectrum'," +
                             "  'significant' or 'non_significant'," +
                             "got {} instead".format(report))
    elif spectrum == '3d#':
        if report == 'spectrum':
            return [(size, supp, l, test)
                    for (size, supp, l, pv), test in zip(
                    pvalue_spectrum, tests)]
        elif report == 'significant':
            return [(size, supp, l) for ((size, supp, l, pv), test)
                    in zip(pvalue_spectrum, tests) if test]
        elif report == 'non_significant':
            return [
                (size, supp, l) for ((size, supp, l, pv), test) in zip(
                    pvalue_spectrum, tests) if not test]
        raise AttributeError("report must be either 'spectrum'," +
                             "  'significant' or 'non_significant'," +
                             "got {} instead".format(report))
    raise AttributeError("spectrum must be either '#' or '3d#', "
                         "got {} instead".format(spectrum))


def _pattern_spectrum_filter(concept, ns_signature, spectrum, winlen):
    """
    Filter to select concept which signature is significant
    """
    if spectrum == '#':
        keep_concept = (len(concept[0]), len(concept[1])) not in ns_signature
    if spectrum == '3d#':
        bin_ids = sorted(np.array(concept[0]) % winlen)
        # The duration is effectively the delay between the last neuron and
        # the first one, measured in bins. Since we only allow the first spike
        # to be in the first bin (see concepts_mining, _build_context and
        # _fpgrowth), it is the the position of the latest spike.
        duration = bin_ids[-1]
        keep_concept = (len(concept[0]), len(concept[1]),
                        duration) not in ns_signature
    return keep_concept


def approximate_stability(concepts, rel_matrix, n_subsets, delta=0, epsilon=0):
    """
    Approximate the stability of concepts. Uses the algorithm described
    in Babin, Kuznetsov (2012): Approximating Concept Stability

    Parameters
    ----------
    concepts: list
        All the pattern candidates (concepts) found in the data. Each
        pattern is represented as a tuple containing (spike IDs,
        discrete times (window position)
        of the  occurrences of the pattern). The spike IDs are defined as:
        spike_id=neuron_id*bin_id; with neuron_id in [0, len(data)] and
        bin_id in [0, winlen].
    rel_matrix: sparse.coo_matrix
        A binary matrix with shape (number of windows, winlen*len(data)). Each
        row corresponds to a window (order according to their position in
        time). Each column corresponds to one bin and one neuron and it is 0 if
        no spikes or 1 if one or more spikes occurred in that bin for that
        particular neuron. For example, the entry [0,0] of this matrix
        corresponds to the first bin of the first window position for the first
        neuron, the entry [0,winlen] to the first bin of the first window
        position for the second neuron.
    n_subsets: int
        Number of subsets of a concept used to approximate its stability. If
        n_subset is set to 0 the stability is not computed. If, however,
        for parameters delta and epsilon (see below) delta + epsilon > 0,
        then an optimal n_subsets is calculated according to the formula given
        in Babin, Kuznetsov (2012), proposition 6:

         ..math::
                n_subset = frac{1}{2\eps^2} \ln(frac{2}{\delta}) +1

        Default:0
    delta: float
        delta: probability with at least ..math:$1-\delta$
        Default: 0
    epsilon: float
        epsilon: absolute error
        Default: 0

    Returns
    -------
    output: list
        List of all the pattern candidates (concepts) given in input, each with
        the correspondent intensional and extensional stability. Each
        pattern is represented as a tuple containing:
         (spike IDs,
        discrete times of the  occurrences of the pattern, intensional
        stability of the pattern, extensional stability of the pattern).
        The spike IDs are defined as:
        spike_id=neuron_id*bin_id; with neuron_id in [0, len(data)] and
        bin_id in [0, winlen].

    Notes
    -----
        If n_subset is larger than the extent all subsets are directly
        calculated, otherwise for small extent size an infinite
        loop can be created while doing the recursion,
        since the random generation will always contain the same
        numbers and the algorithm will be stuck searching for
        other (random) numbers

    """
    if HAVE_MPI:  # pragma: no cover
        comm = MPI.COMM_WORLD  # create MPI communicator
        rank = comm.Get_rank()  # get rank of current MPI task
        size = comm.Get_size()  # get tot number of MPI tasks
    else:
        rank = 0
        size = 1
    if n_subsets <= 0 and delta + epsilon <= 0:
        raise AttributeError('n_subsets has to be >=0 or delta + epsilon > 0')
    if len(concepts) == 0:
        return []
    elif len(concepts) <= size:
        rank_idx = [0] * (size + 1) + [len(concepts)]
    else:
        rank_idx = list(
            np.arange(
                0, len(concepts) - len(concepts) % size + 1,
                   len(concepts) // size)) + [len(concepts)]
    # Calculate optimal n
    if delta + epsilon > 0 and n_subsets == 0:
        n_subsets = np.log(2. / delta) / (2 * epsilon ** 2) + 1
    output = []
    if rank == 0:
        for concept in concepts[
                       rank_idx[rank]:rank_idx[rank + 1]] + concepts[
                                                            rank_idx[-2]:
                                                            rank_idx[-1]]:
            stab_ext = 0.0
            stab_int = 0.0
            intent = np.array(list(concept[0]))
            extent = np.array(list(concept[1]))
            r_unique_ext = set()
            r_unique_int = set()
            excluded_subset = []
            # Calculate all subsets if n is larger than the power set of
            # the extent
            if n_subsets > 2 ** len(extent):
                subsets_ext = chain.from_iterable(
                    combinations(extent, r) for r in range(
                        len(extent) + 1))
                for s in subsets_ext:
                    if any(
                            [set(s).issubset(se) for se in excluded_subset]):
                        continue
                    if _closure_probability_extensional(
                            intent, s, rel_matrix):
                        stab_ext += 1
                    else:
                        excluded_subset.append(s)
            else:
                for _ in range(n_subsets):
                    subset_ext = extent[
                        _give_random_idx(r_unique_ext, len(extent))]
                    if any([
                        set(subset_ext).issubset(se) for
                        se in excluded_subset]):
                        continue
                    if _closure_probability_extensional(
                            intent, subset_ext, rel_matrix):
                        stab_ext += 1
                    else:
                        excluded_subset.append(subset_ext)
            stab_ext /= min(n_subsets, 2 ** len(extent))
            excluded_subset = []
            # Calculate all subsets if n is larger than the power set of
            # the extent
            if n_subsets > 2 ** len(intent):
                subsets_int = chain.from_iterable(
                    combinations(intent, r) for r in range(
                        len(intent) + 1))
                for s in subsets_int:
                    if any(
                            [set(s).issubset(se) for se in excluded_subset]):
                        continue
                    if _closure_probability_intensional(
                            extent, s, rel_matrix):
                        stab_int += 1
                    else:
                        excluded_subset.append(s)
            else:
                for _ in range(n_subsets):
                    subset_int = intent[
                        _give_random_idx(r_unique_int, len(intent))]
                    if any([
                        set(subset_int).issubset(se) for
                        se in excluded_subset]):
                        continue
                    if _closure_probability_intensional(
                            extent, subset_int, rel_matrix):
                        stab_int += 1
                    else:
                        excluded_subset.append(subset_int)
            stab_int /= min(n_subsets, 2 ** len(intent))
            output.append((intent, extent, stab_int, stab_ext))
    else:  # pragma: no cover
        for concept in concepts[rank_idx[rank]:rank_idx[rank + 1]]:
            stab_ext = 0.0
            stab_int = 0.0
            intent = np.array(list(concept[0]))
            extent = np.array(list(concept[1]))
            r_unique_ext = set()
            r_unique_int = set()
            excluded_subset = []
            # Calculate all subsets if n is larger than the power set of
            # the extent
            if n_subsets > 2 ** len(extent):
                subsets_ext = chain.from_iterable(
                    combinations(extent, r) for r in range(
                        len(extent) + 1))
                for s in subsets_ext:
                    if any(
                            [set(s).issubset(se) for se in excluded_subset]):
                        continue
                    if _closure_probability_extensional(
                            intent, s, rel_matrix):
                        stab_ext += 1
                    else:
                        excluded_subset.append(s)
            else:
                for _ in range(n_subsets):
                    subset_ext = extent[
                        _give_random_idx(r_unique_ext, len(extent))]
                    if any([
                        set(subset_ext).issubset(se) for
                        se in excluded_subset]):
                        continue
                    if _closure_probability_extensional(
                            intent, subset_ext, rel_matrix):
                        stab_ext += 1
                    else:
                        excluded_subset.append(subset_ext)
            stab_ext /= min(n_subsets, 2 ** len(extent))
            excluded_subset = []
            # Calculate all subsets if n is larger than the power set of
            # the extent
            if n_subsets > 2 ** len(intent):
                subsets_int = chain.from_iterable(
                    combinations(intent, r) for r in range(
                        len(intent) + 1))
                for s in subsets_int:
                    if any(
                            [set(s).issubset(se) for se in excluded_subset]):
                        continue
                    if _closure_probability_intensional(
                            extent, s, rel_matrix):
                        stab_int += 1
                    else:
                        excluded_subset.append(s)
            else:
                for _ in range(n_subsets):
                    subset_int = intent[
                        _give_random_idx(r_unique_int, len(intent))]
                    if any([
                        set(subset_int).issubset(se) for
                        se in excluded_subset]):
                        continue
                    if _closure_probability_intensional(
                            extent, subset_int, rel_matrix):
                        stab_int += 1
                    else:
                        excluded_subset.append(subset_int)
            stab_int /= min(n_subsets, 2 ** len(intent))
            output.append((intent, extent, stab_int, stab_ext))

    if rank != 0:  # pragma: no cover
        comm.send(output, dest=0)
    if rank == 0:  # pragma: no cover
        for i in range(1, size):
            recv_list = comm.recv(source=i)
            output.extend(recv_list)

    return output


def _closure_probability_extensional(intent, subset, rel_matrix):
    """
    Return True if the closure of the subset of the extent given in input is
    equal to the intent given in input

    Parameters
    ----------
    intent : array
    Set of the attributes of the concept
    subset : list
    List of objects that form the subset of the extent to be evaluated
    rel_matrix: ndarray
    Binary matrix that specify the relation that defines the context

    Returns
    -------
    1 if (subset)' == intent
    0 else
    """
    # computation of the ' operator for the subset
    subset_prime = np.where(np.all(rel_matrix[subset, :], axis=0) == 1)[0]
    if set(subset_prime) == set(list(intent)):
        return 1
    return 0


def _closure_probability_intensional(extent, subset, rel_matrix):
    """
    Return True if the closure of the subset of the intent given in input is
    equal to the extent given in input

    Parameters
    ----------
    extent : list
    Set of the objects of the concept
    subset : list
    List of attributes that form the subset of the intent to be evaluated
    rel_matrix: ndarray
    Binary matrix that specify the relation that defines the context

    Returns
    -------
    1 if (subset)' == extent
    0 else
    """
    # computation of the ' operator for the subset
    subset_prime = np.where(np.all(rel_matrix[:, subset], axis=1) == 1)[0]
    if set(subset_prime) == set(list(extent)):
        return 1
    return 0


def _give_random_idx(r_unique, n):
    """ asd """

    r = np.random.randint(n,
                          size=np.random.randint(low=1,
                                                 high=n))
    r_tuple = tuple(r)
    if r_tuple not in r_unique:
        r_unique.add(r_tuple)
        return np.unique(r)
    return _give_random_idx(r_unique, n)


def pattern_set_reduction(concepts, excluded, winlen, h=0, k=0, l=0,
                          min_spikes=2, min_occ=2):
    """
    Takes a list concepts and performs pattern set reduction (PSR).

    PSR determines which patterns in concepts_psf are statistically significant
    given any other pattern, on the basis of the pattern size and
    occurrence count ("support"). Only significant patterns are retained.
    The significance of a pattern A is evaluated through its signature
    (|A|,c_A), where |A| is the size and c_A the support of A, by either of:
    * subset filtering: any pattern B is discarded if *cfis* contains a
      superset A of B such that (z_B, c_B-c_A+*h*) \in *excluded*
    * superset filtering: any pattern A is discarded if *cfis* contains a
      subset B of A such that (z_A-z_B+*k*, c_A) \in  *excluded*
    * covered-spikes criterion: for any two patterns A, B with A \subset B, B
      is discarded if (z_B-l)*c_B <= c_A*(z_A-*l*), A is discarded otherwise.
    * combined filtering: combines the three procedures above
    takes a list concepts (see output psf function) and performs
    combined filtering based on the signature (z, c) of each pattern, where
    z is the pattern size and c the pattern support.

    For any two patterns A and B in concepts_psf such that B \subset A, check:
    1) (z_B, c_B-c_A+*h*) \in *excluded*, and
    2) (z_A-z_B+*k*, c_A) \in *excluded*.
    Then:
    * if 1) and not 2): discard B
    * if 2) and not 1): discard A
    * if 1) and 2): discard B if c_B*(z_B-*l*) <= c_A*(z_A-*l*), A otherwise;
    * if neither 1) nor 2): keep both patterns.

    Parameters:
    -----------
    concept: list
        List of concepts, each consisting in its intent and extent
    excluded: list
        A list of non-significant pattern signatures (z, c) (see above).
    h: int
        Correction parameter for subset filtering (see above).
        Defaults: 0
    k: int
        Correction parameter for superset filtering (see above).
        Default: 0
    l: int ]
        Correction parameter for covered-spikes criterion (see above).
        Default: 0
    min_size: int
        Minimum pattern size.
        Default: 2
    min_supp: int
        Minimum pattern support.
        Default: 2

    Returns:
    -------
      returns a tuple containing the elements of the input argument
      that are significant according to combined filtering.
    """
    conc = []
    # Extracting from the extent and intent the spike and window times
    for concept in concepts:
        intent = concept[0]
        extent = concept[1]
        spike_times = np.array([st % winlen for st in intent])
        conc.append((intent, spike_times, extent, len(extent)))

    # by default, select all elements in conc to be returned in the output
    selected = [True for _ in conc]
    # scan all conc and their subsets
    for id1, (conc1, s_times1, winds1, count1) in enumerate(conc):
        for id2, (conc2, s_times2, winds2, count2) in enumerate(conc):
            if not selected[id1]:
                break
            if id1 == id2:
                continue
            # Collecting all the possible distances between the windows
            # of the two concepts
            time_diff_all = np.array(
                [w2 - w1 for w2 in winds2 for w1 in winds1])
            sorted_time_diff = np.unique(
                time_diff_all[np.argsort(np.abs(time_diff_all))])
            # Rescaling the spike times to realign to real time
            for time_diff in sorted_time_diff[
                np.abs(sorted_time_diff) < winlen]:
                conc1_new = [
                    t_old - time_diff for t_old in conc1]
                # if conc1 is  of conc2 are disjoint or they have both been
                # already de-selected, skip the step
                if set(conc1_new) == set(
                        conc2) and selected[id1] and selected[id2]:
                    selected[id2] = False
                    break
                if not set(conc1_new) & set(conc2):
                    continue
                if not selected[id1]:
                    continue
                if not selected[id2]:
                    continue
                if set(conc1_new).issuperset(conc2) and count2 \
                        - count1 + h < min_occ:
                    selected[id2] = False
                    break
                if set(conc2).issuperset(conc1_new) and count1 \
                        - count2 + h < min_occ:
                    selected[id1] = False
                    break
                if len(excluded) == 0:
                    break
                # Test the case conc1 is a superset of conc2
                if set(conc1_new).issuperset(conc2):
                    supp_diff = count2 - count1 + h
                    size1, size2 = len(conc1_new), len(conc2)
                    size_diff = size1 - size2 + k
                    # 2d spectrum case
                    if len(excluded[0]) == 2:
                        # Determine whether the subset (conc2)
                        # should be rejected
                        # according to the test for excess occurrences
                        reject_sub = (size2, supp_diff) in excluded \
                                     or supp_diff < min_occ
                        # Determine whether the superset (conc1_new) should be
                        # rejected according to the test for excess items
                        reject_sup = (size_diff, count1) in excluded \
                                     or size_diff < min_spikes
                    # 3d spectrum case
                    if len(excluded[0]) == 3:
                        # Determine whether the subset (conc2)
                        # should be rejected
                        # according to the test for excess occurrences
                        len_sub = max(
                            np.abs(np.diff(np.array(conc2) % winlen)))
                        reject_sub = (size2, supp_diff, len_sub) in excluded \
                                     or supp_diff < min_occ
                        # Determine whether the superset (conc1_new) should be
                        # rejected according to the test for excess items
                        len_sup = max(
                            np.abs(np.diff(np.array(conc1_new) % winlen)))
                        reject_sup = (size_diff, count1, len_sup) in excluded \
                                     or size_diff < min_spikes
                    # Reject the superset and/or the subset accordingly:
                    if reject_sub and not reject_sup:
                        selected[id2] = False
                        break
                    elif reject_sup and not reject_sub:
                        selected[id1] = False
                        break
                    elif reject_sub and reject_sup:
                        if (size1 - l) * count1 >= (size2 - l) * count2:
                            selected[id2] = False
                            break
                        else:
                            selected[id1] = False
                            break
                    # if both sets are significant given the other, keep both
                    else:
                        continue
                elif set(conc2).issuperset(conc1_new):
                    supp_diff = count1 - count2 + h
                    size1, size2 = len(conc1_new), len(conc2)
                    size_diff = size2 - size1 + k
                    # 2d spectrum case
                    if len(excluded[0]) == 2:
                        # Determine whether the subset (conc2) should be
                        # rejected according to the test for excess occurrences
                        reject_sub = (size2, supp_diff) in excluded \
                                     or supp_diff < min_occ
                        # Determine whether the superset (conc1_new) should be
                        # rejected according to the test for excess items
                        reject_sup = (size_diff, count1) in excluded \
                                     or size_diff < min_spikes
                    # 3d spectrum case
                    if len(excluded[0]) == 3:
                        # Determine whether the subset (conc2) should be
                        # rejected according to the test for excess occurrences
                        len_sub = max(
                            np.abs(np.diff(np.array(conc1) % winlen)))
                        reject_sub = (size2, supp_diff, len_sub) in excluded \
                                     or supp_diff < min_occ
                        # Determine whether the superset (conc1_new) should be
                        # rejected according to the test for excess items
                        len_sup = max(
                            np.abs(np.diff(np.array(conc2) % winlen)))

                        reject_sup = (size_diff, count1, len_sup) in excluded \
                                     or size_diff < min_spikes
                    # Reject the superset and/or the subset accordingly:
                    if reject_sub and not reject_sup:
                        selected[id1] = False
                        break
                    elif reject_sup and not reject_sub:
                        selected[id2] = False
                        break
                    elif reject_sub and reject_sup:
                        if (size1 - l) * count1 >= (size2 - l) * count2:
                            selected[id2] = False
                            break
                        else:
                            selected[id1] = False
                            break
                    # if both sets are significant given the other, keep both
                    else:
                        continue
                else:
                    size1, size2 = len(conc1_new), len(conc2)
                    inter_size = len(set(conc1_new) & set(conc2))
                    # 2d spectrum case
                    if len(excluded[0]) == 2:
                        reject_1 = (
                                       size1 - inter_size + k,
                                       count1) in \
                                   excluded or size1 - inter_size + k < min_spikes
                        reject_2 = (
                                       size2 - inter_size + k,
                                       count2) in excluded or \
                                   size2 - inter_size + k < min_spikes
                    # 3d spectrum case
                    if len(excluded[0]) == 3:
                        len_1 = max(
                            np.abs(np.diff(np.array(conc1_new) % winlen)))
                        len_2 = max(
                            np.abs(np.diff(np.array(conc2) % winlen)))
                        reject_1 = (
                                       size1 - inter_size + k, count1,
                                       len_1) in excluded or \
                                   size1 - inter_size + k < min_spikes
                        reject_2 = (
                                       size2 - inter_size + k, count2, len_2) \
                                   in excluded or \
                                   size2 - inter_size + k < min_spikes

                    # Reject accordingly:
                    if reject_2 and not reject_1:
                        selected[id2] = False
                        break
                    elif reject_1 and not reject_2:
                        selected[id1] = False
                        break
                    elif reject_1 and reject_2:
                        if (size1 - l) * count1 >= (size2 - l) * count2:
                            selected[id2] = False
                            break
                        else:
                            selected[id1] = False
                            break
                    # if both sets are significant given the other, keep both
                    else:
                        continue

    # Return the selected concepts
    return [p for i, p in enumerate(concepts) if selected[i]]


def concept_output_to_patterns(concepts, winlen, binsize, pvalue_spectrum=None,
                               spectrum=None, t_start=0 * pq.ms):
    """
    Construction of dictionaries containing all the information about a pattern
    starting from a list of concepts and its associated pvalue_spectrum.

    Parameters
    ----------
    concepts: tuple
        Each element of the tuple corresponds to a pattern and it is itself a
        tuple consisting of:
            ((spikes in the pattern), (occurrences of the patterns))
    winlen: int
        Length (in bins) of the sliding window used for the analysis
    binsize: Quantity
        The time precision used to discretize the data (binning).
    pvalue_spectrum: None or tuple
        Contains a tuple of signatures and the corresponding p-value. If equal
        to None all pvalues are set to -1
    spectrum: None or str
        The signature of the given concepts, it can assume values:
        None: the signature is determined from the pvalue_spectrum
        '#': pattern spectrum using the as signature the pair:
            (number of spikes, number of occurrences)
        '3d#': pattern spectrum using the as signature the triplets:
            (number of spikes, number of occurrence, difference between last
            and first spike of the pattern)
        Default: None
    t_start: Quantity
        t_start of the analyzed spike trains

    Returns
    --------
    output: list
        List of dictionaries. Each dictionary corresponds to a pattern and
        has the following entries:
            ['itemset'] list of the spikes in the pattern
                expressed in the form of itemset, each spike is encoded by:
                spiketrain_id * winlen + bin_id
            [windows_ids'] the ids of the windows in which the pattern occurred
                in discretized time (given byt the binning)
            ['neurons'] array containing the idx of the neurons of the pattern
            ['lags'] array containing the lags (integers corresponding to the
                number of bins) between the spikes of the patterns. The first
                lag is always assumed to be 0 and corresponds to the first
                spike.
            ['times'] array containing the times (integers corresponding to the
                bin idx) of the occurrences of the patterns.
            ['signature'] tuple containing two integers
                (number of spikes of the patterns,
                number of occurrences of the pattern)
            ['pvalue'] the pvalue corresponding to the pattern. If n_surr==0
             then all pvalues are set to -1.
    """
    if pvalue_spectrum is None:
        if spectrum is None:
            spectrum = '#'
    else:
        if spectrum is None:
            if len(pvalue_spectrum) == 0:
                spectrum = '#'
            elif len(pvalue_spectrum[0]) == 4:
                spectrum = '3d#'
            elif len(pvalue_spectrum[0]) == 3:
                spectrum = '#'
        pvalue_dict = {}
        # Creating a dictionary for the pvalue spectrum
        for entry in pvalue_spectrum:
            if len(entry) == 4:
                pvalue_dict[(entry[0], entry[1], entry[2])] = entry[-1]
            if len(entry) == 3:
                pvalue_dict[(entry[0], entry[1])] = entry[-1]
    # Initializing list containing all the patterns
    output = []
    for conc in concepts:
        # Vocabulary for each of the patterns, containing:
        # - The pattern expressed in form of Itemset, each spike in the pattern
        # is represented as spiketrain_id * winlen + bin_id
        # - The ids of the windows in which the pattern occurred in discretized
        # time (binning)
        output_dict = {'itemset': conc[0], 'windows_ids': conc[1]}
        # Bins relative to the sliding window in which the spikes of patt fall
        bin_ids_unsort = np.array(conc[0]) % winlen
        order_bin_ids = np.argsort(bin_ids_unsort)
        bin_ids = bin_ids_unsort[order_bin_ids]
        # id of the neurons forming the pattern
        output_dict['neurons'] = list(np.array(
            conc[0])[order_bin_ids] // winlen)
        # Lags (in binsizes units) of the pattern
        output_dict['lags'] = (bin_ids - bin_ids[0])[1:] * binsize
        # Times (in binsize units) in which the pattern occurs
        output_dict['times'] = sorted(conc[1]) * binsize + bin_ids[0] * \
                               binsize + t_start
        # If None is given in input to the pval spectrum the pvalue
        # is set to -1 (pvalue spectrum not available)
        # pattern dictionary appended to the output
        if pvalue_spectrum is None:
            output_dict['pvalue'] = -1
        # Signature (size, n occ) of the pattern
        elif spectrum == '3d#':
            # The duration is effectively the delay between the last neuron and
            # the first one, measured in bins.
            # Since we only allow the first spike
            # to be in the first bin (see concepts_mining, _build_context and
            # _fpgrowth), it is the the position of the latest spike.
            duration = bin_ids[-1]
            sgnt = (len(conc[0]), len(conc[1]), duration)
            output_dict['signature'] = sgnt
            # p-value assigned to the pattern from the pvalue spectrum
            try:
                output_dict['pvalue'] = pvalue_dict[sgnt]
            except KeyError:
                output_dict['pvalue'] = 0.0
        elif spectrum == '#':
            sgnt = (len(conc[0]), len(conc[1]))
            output_dict['signature'] = sgnt
            # p-value assigned to the pattern from the pvalue spectrum
            try:
                output_dict['pvalue'] = pvalue_dict[sgnt]
            except KeyError:
                output_dict['pvalue'] = 0.0
        output.append(output_dict)
    return output
