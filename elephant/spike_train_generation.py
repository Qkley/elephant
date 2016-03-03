"""
Functions to generate spike trains from analog signals,
or to generate random spike trains.

Most of these functions were adapted from the NeuroTools stgen module,
which was mostly written by Eilif Muller,
or from the NeuroTools signals.analogs module.

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division
import numpy as np
from quantities import ms, mV, Hz, Quantity, dimensionless
from neo import SpikeTrain
import random
from elephant.spike_train_surrogates import dither_spike_train

def threshold_detection(signal, threshold=0.0*mV, sign='above'):
    """
    Returns the times when the analog signal crosses a threshold.
    Usually used for extracting spike times from a membrane potential.
    Adapted from version in NeuroTools.

    Parameters
    ----------
    signal : neo AnalogSignal object
        'signal' is an analog signal.
    threshold : A quantity, e.g. in mV
        'threshold' contains a value that must be reached
        for an event to be detected.
    sign : 'above' or 'below'
        'sign' determines whether to count thresholding crossings
        that cross above or below the threshold.
    format : None or 'raw'
        Whether to return as SpikeTrain (None)
        or as a plain array of times ('raw').

    Returns
    -------
    result_st : neo SpikeTrain object
        'result_st' contains the spike times of each of the events (spikes)
        extracted from the signal.
    """

    assert threshold is not None, "A threshold must be provided"

    if sign is 'above':
        cutout = np.where(signal > threshold)[0]
    elif sign in 'below':
        cutout = np.where(signal < threshold)[0]

    if len(cutout) <= 0:
        events = np.zeros(0)
    else:
        take = np.where(np.diff(cutout)>1)[0]+1
        take = np.append(0,take)

        time = signal.times
        events = time[cutout][take]

    events_base = events.base
    if events_base is None: # This occurs in some Python 3 builds due to some
                            # bug in quantities.
        events_base = np.array([event.base for event in events]) # Workaround

    result_st = SpikeTrain(events_base,units=signal.times.units,
                           t_start=signal.t_start,t_stop=signal.t_stop)
    return result_st


def _homogeneous_process(interval_generator, args, mean_rate, t_start, t_stop, as_array):
    """
    Returns a spike train whose spikes are a realization of a random process generated by the function `interval_generator`
    with the given rate, starting at time `t_start` and stopping `time t_stop`.
    """
    def rescale(x):
        return (x / mean_rate.units).rescale(t_stop.units)

    n = int(((t_stop - t_start) * mean_rate).simplified)
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2 * n), 100)
    assert number > 4  # if positive, number cannot be less than 5
    isi = rescale(interval_generator(*args, size=number))
    spikes = np.cumsum(isi)
    spikes += t_start

    i = spikes.searchsorted(t_stop)
    if i == len(spikes):
        # ISI buffer overrun
        extra_spikes = []
        t_last = spikes[-1] + rescale(interval_generator(*args, size=1))[0]
        while t_last < t_stop:
            extra_spikes.append(t_last)
            t_last = t_last + rescale(interval_generator(*args, size=1))[0]
        # np.concatenate does not conserve units
        spikes = Quantity(
            np.concatenate((spikes, extra_spikes)).magnitude, units=spikes.units)
    else:
        spikes = spikes[:i]

    if as_array:
        spikes = spikes.magnitude
    else:
        spikes = SpikeTrain(
            spikes, t_start=t_start, t_stop=t_stop, units=spikes.units)

    return spikes


def homogeneous_poisson_process(rate, t_start=0.0 * ms, t_stop=1000.0 * ms, as_array=False):
    """
    Returns a spike train whose spikes are a realization of a Poisson process
    with the given rate, starting at time `t_start` and stopping time `t_stop`.

    All numerical values should be given as Quantities, e.g. 100*Hz.

    Parameters
    ----------

    rate : Quantity scalar with dimension 1/time
           The rate of the discharge.
    t_start : Quantity scalar with dimension time
              The beginning of the spike train.
    t_stop : Quantity scalar with dimension time
             The end of the spike train.
    as_array : bool
               If True, a NumPy array of sorted spikes is returned,
               rather than a SpikeTrain object.

    Examples
    --------
        >>> from quantities import Hz, ms
        >>> spikes = homogeneous_poisson_process(50*Hz, 0*ms, 1000*ms)
        >>> spikes = homogeneous_poisson_process(20*Hz, 5000*ms, 10000*ms, as_array=True)

    """
    mean_interval = 1 / rate
    return _homogeneous_process(np.random.exponential, (mean_interval,), rate, t_start, t_stop, as_array)


def homogeneous_gamma_process(a, b, t_start=0.0 * ms, t_stop=1000.0 * ms, as_array=False):
    """
    Returns a spike train whose spikes are a realization of a gamma process
    with the given parameters, starting at time `t_start` and stopping time `t_stop`.
    (average rate will be b/a).

    All numerical values should be given as Quantities, e.g. 100*Hz.

    Parameters
    ----------

    a : int or float
        The shape parameter of the gamma distribution.
    b : Quantity scalar with dimension 1/time
        The rate parameter of the gamma distribution.
    t_start : Quantity scalar with dimension time
              The beginning of the spike train.
    t_stop : Quantity scalar with dimension time
             The end of the spike train.
    as_array : bool
               If True, a NumPy array of sorted spikes is returned,
               rather than a SpikeTrain object.

    Examples
    --------
        >>> from quantities import Hz, ms
        >>> spikes = homogeneous_gamma_process(2.0, 50*Hz, 0*ms, 1000*ms)
        >>> spikes = homogeneous_gamma_process(5.0, 20*Hz, 5000*ms, 10000*ms, as_array=True)

    """
    rate = b / a
    k, theta = a, (1 / b)
    return _homogeneous_process(np.random.gamma, (k, theta), rate, t_start, t_stop, as_array)

def _n_poisson(rate, t_stop, t_start=0.0 * ms, n=1):
    """
    Generates one or more independent Poisson spike trains.
    Parameters
    ----------
    rate : Quantity or Quantity array
        Expected firing rate (frequency) of each output SpikeTrain.
        Can be one of:
        *  a single Quantity value: expected firing rate of each output
           SpikeTrain
        *  a Quantity array: rate[i] is the expected firing rate of the i-th
           output SpikeTrain
    t_stop : Quantity
        Single common stop time of each output SpikeTrain. Must be > t_start.
    t_start : Quantity (optional)
        Single common start time of each output SpikeTrain. Must be < t_stop.
        Default: 0 s.
    n: int (optional)
        If rate is a single Quantity value, n specifies the number of
        SpikeTrains to be generated. If rate is an array, n is ignored and the
        number of SpikeTrains is equal to len(rate).
        Default: 1
    Returns
    -------
    list of neo.SpikeTrain
        Each SpikeTrain contains one of the independent Poisson spike trains,
        either n SpikeTrains of the same rate, or len(rate) SpikeTrains with
        varying rates according to the rate parameter. The time unit of the
        SpikeTrains is given by t_stop.
    """
    # Check that the provided input is Hertz of return error
    try:
        for r in rate.reshape(-1, 1):
            r.rescale('Hz')
    except AttributeError:
        raise ValueError('rate argument must have rate unit (1/time)')

    # Check t_start < t_stop and create their strip dimensions
    if not t_start < t_stop:
        raise ValueError(
            't_start (=%s) must be < t_stop (=%s)' % (t_start, t_stop))

    # Set number n of output spike trains (specified or set to len(rate))
    if not (type(n) == int and n > 0):
        raise ValueError('n (=%s) must be a positive integer' % str(n))
    rate_dl = rate.simplified.magnitude.flatten()

    # Check rate input parameter
    if len(rate_dl) == 1:
        if rate_dl < 0:
            raise ValueError('rate (=%s) must be non-negative.' % rate)
        rates = np.array([rate_dl] * n)
    else:
        rates = rate_dl.flatten()
        if any(rates < 0):
            raise ValueError('rate must have non-negative elements.')
    sts = []
    for r in rates:
        sts.append(homogeneous_poisson_process(r*Hz, t_start, t_stop))
    return sts


def _pool_two_spiketrains(a, b, range='inner'):
    """
    Pool the spikes of two spike trains a and b into a unique spike train.

    Parameters
    ----------
    a, b : neo.SpikeTrains
        Spike trains to be pooled

    range: str, optional
        Only spikes of a and b in the specified range are considered.
        * 'inner': pool all spikes from min(a.tstart_ b.t_start) to
           max(a.t_stop, b.t_stop)
        * 'outer': pool all spikes from max(a.tstart_ b.t_start) to
           min(a.t_stop, b.t_stop)
        Default: 'inner'

    Output
    ------
    neo.SpikeTrain containing all spikes in a and b falling in the
    specified range
    """

    unit = a.units
    times_a_dimless = list(a.view(Quantity).magnitude)
    times_b_dimless = list(b.rescale(unit).view(Quantity).magnitude)
    times = (times_a_dimless + times_b_dimless) * unit

    if range == 'outer':
        t_start = min(a.t_start, b.t_start)
        stop = max(a.t_stop, b.t_stop)
        times = times[times > t_start]
        times = times[times < stop]
    elif range == 'inner':
        t_start = max(a.t_start, b.t_start)
        stop = min(a.t_stop, b.t_stop)
    else:
        raise ValueError('range (%s) can only be "inner" or "outer"' % range)
    pooled_train = SpikeTrain(
        times=sorted(times.magnitude), units=unit, t_start=t_start, t_stop=stop)
    return pooled_train


def _pool_spiketrains(trains, range='inner'):
    """
    Pool spikes from any number of spike trains into a unique spike train.

    Parameters
    ----------
    trains [list]
        list of spike trains to merge

    range: str, optional
        Only spikes of a and b in the specified range are considered.
        * 'inner': pool all spikes from min(a.t_start b.t_start) to
           max(a.t_stop, b.t_stop)
        * 'outer': pool all spikes from max(a.tstart_ b.t_start) to
           min(a.t_stop, b.t_stop)
        Default: 'inner'

    Output
    ------
    neo.SpikeTrain containing all spikes in trains falling in the
    specified range
    """

    merge_trains = trains[0]
    for t in trains[1:]:
        merge_trains = _pool_two_spiketrains(merge_trains, t, range=range)
    t_start, stop = merge_trains.t_start, merge_trains.t_stop
    merge_trains = sorted(merge_trains)
    merge_trains = np.squeeze(merge_trains)
    merge_trains = SpikeTrain(
        merge_trains, t_stop=stop, t_start=t_start, units=trains[0].units)
    return merge_trains


def _sample_int_from_pdf(a, n):
    """
    Draw n independent samples from the set {0,1,...,L}, where L=len(a)-1,
    according to the probability distribution a.
    a[j] is the probability to sample j, for each j from 0 to L.


    Parameters
    -----
    a [array|list]
        Probability vector (i..e array of sum 1) that at each entry j carries
        the probability to sample j (j=0,1,...,len(a)-1).

    n [int]
        Number of samples generated with the function

    Output
    -------
    array of n samples taking values between 0 and n=len(a)-1.
    """

    # a = np.array(a)
    A = np.cumsum(a)  # cumulative distribution of a
    u = np.random.uniform(0, 1, size=n)
    U = np.array([u for i in a]).T  # copy u (as column vector) len(a) times
    return (A < U).sum(axis=1)


def _mother_proc_cpp_stat(A, t_stop, r, t_start=0 * ms):
    """
    Generate the hidden ("mother") Poisson process for a Compound Poisson
    Process (CPP).


    Parameters
    ----------
    r : Quantity, Hz
        Homogeneous rate of the n spike trains that will be genereted by the
        CPP function
    A : array
        Amplitude distribution. A[j] represents the probability of a
        synchronous event of size j.
        The sum over all entries of a must be equal to one.
    t_stop : Quantity (time)
        The stopping time of the mother process
    t_start : Quantity (time). Optional, default is 0 ms
        The starting time of the mother process


    Output
    ------
    Poisson spike train representing the mother process generating the CPP
    """

    N = len(A) - 1
    exp_A = np.dot(A, range(N + 1))  # expected value of a
    exp_mother = (N * r) / float(exp_A)  # rate of the mother process
    return homogeneous_poisson_process(
        rate=exp_mother, t_stop=t_stop, t_start=t_start)


def _cpp_hom_stat(A, t_stop, r, t_start=0 * ms):
    """
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and heterogeneous firing rates r=r[0], r[1], ..., r[-1].

    Parameters
    ----------
    A : array
        Amplitude distribution. A[j] represents the probability of a
        synchronous event of size j.
        The sum over all entries of A must be equal to one.
    t_stop : Quantity (time)
        The end time of the output spike trains
    r : Quantity (1/time)
        Average rate of each spike train generated
    t_start : Quantity (time). Optional, default to 0 s
        The start time of the output spike trains

    Output
    ------
    List of n neo.SpikeTrains, having average firing rate r and correlated
    such to form a CPP with amplitude distribution a
    """

    # Generate mother process and associated spike labels
    mother = _mother_proc_cpp_stat(A=A, t_stop=t_stop, r=r, t_start=t_start)
    labels = _sample_int_from_pdf(A, len(mother))

    N = len(A) - 1  # Number of trains in output

    try:  # Faster but more memory-consuming approach
        M = len(mother)  # number of spikes in the mother process
        spike_matrix = np.zeros((N, M), dtype=bool)
        # for each spike, take its label l
        for spike_id, l in enumerate(labels):
            # choose l random trains
            train_ids = random.sample(xrange(N), l)
            # and set the spike matrix for that train
            for train_id in train_ids:
                spike_matrix[train_id, spike_id] = True  # and spike to True

        times = [[] for i in range(N)]
        for train_id, row in enumerate(spike_matrix):
            times[train_id] = mother[row].view(Quantity)

    except MemoryError:  # Slower (~2x) but less memory-consuming approach
        print 'memory case'
        times = [[] for i in range(N)]
        for t, l in zip(mother, labels):
            train_ids = random.sample(xrange(N), l)
            for train_id in train_ids:
                times[train_id].append(t)

    trains = [SpikeTrain(
        times=t, t_start=t_start, t_stop=t_stop) for t in times]

    return trains


def _cpp_het_stat(A, t_stop, r, t_start=0.*ms):
    """
    Generate a Compound Poisson Process (CPP) with amplitude distribution
    A and heterogeneous firing rates r=r[0], r[1], ..., r[-1].

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    t_stop : Quantity (time)
        The end time of the output spike trains
    r : Quantity (1/time)
        Average rate of each spike train generated
    t_start : Quantity (time). Optional, default to 0 s
        The start time of the output spike trains

    Output
    ------
    List of neo.SpikeTrains with different firing rates, forming
    a CPP with amplitude distribution A
    """

    # Computation of Parameters of the two CPPs that will be merged
    # (uncorrelated with heterog. rates + correlated with homog. rates)
    N = len(r)  # number of output spike trains
    A_exp = np.dot(A, xrange(N + 1))  # expectation of A
    r_sum = np.sum(r)  # sum of all output firing rates
    r_min = np.min(r)  # minimum of the firing rates
    r1 = r_sum - N * r_min  # rate of the uncorrelated CPP
    r2 = r_sum / float(A_exp) - r1  # rate of the correlated CPP
    r_mother = r1 + r2  # rate of the hidden mother process

    # Check the analytical constraint for the amplitude distribution
    if A[1] < (r1 / r_mother).rescale(dimensionless).magnitude:
        raise ValueError('A[1] too small / A[i], i>1 too high')

    # Compute the amplitude distrib of the correlated CPP, and generate it
    a = [(r_mother * i) / float(r2) for i in A]
    a[1] = a[1] - r1 / float(r2)
    CPP = _cpp_hom_stat(a, t_stop, r_min, t_start)

    # Generate the independent heterogeneous Poisson processes
    POISS = [_n_poisson(i - r_min, t_stop, t_start)[0] for i in r]

    # Pool the correlated CPP and the corresponding Poisson processes
    out = [_pool_two_spiketrains(CPP[i], POISS[i]) for i in range(N)]
    return out


def cpp(A, t_stop, rate, t_start=0 * ms, jitter=None):
    """
    Generate a Compound Poisson Process (CPP) with a given amplitude
    distribution A and stationary marginal rates r.

    The CPP process is a model for parallel, correlated processes with Poisson
    spiking statistics at pre-defined firing rates. It is composed of len(A)-1
    spike trains with a correlation structure determined by the amplitude
    distribution A: A[j] is the probability that a spike occurs synchronously
    in any j spike trains.

    The CPP is generated by creating a hidden mother Poisson process, and then
    copying spikes of the mother process to j of the output spike trains with
    probability A[j].

    Note that this function decorrelates the firing rate of each SpikeTrain
    from the probability for that SpikeTrain to participate in a synchronous
    event (which is uniform across SpikeTrains).

    Parameters
    ----------
    A : array
        CPP's amplitude distribution. A[j] represents the probability of
        a synchronous event of size j among the generated spike trains.
        The sum over all entries of A must be equal to one.
    t_stop : Quantity (time)
        The end time of the output spike trains
    rate : Quantity (1/time)
        Average rate of each spike train generated. Can be:
        * single-valued: if so, all spike trains will have same rate rate
        * a sequence of values (of length len(A)-1), each indicating the
          firing rate of one process in output
    t_start : Quantity (time). Optional, default to 0 s
        The t_start time of the output spike trains
    jitter : None or Quantity
        If None the corelations are perfectly synchronous, in the case jitter
        is a quantity object all the spike trains are shifted of a random in
        the interval [-jitter, +jitter].
        Default: None

    Returns
    -------
    List of SpikeTrain
        SpikeTrains with specified firing rates forming the CPP with amplitude
        distribution A.
    """
    if abs(sum(A)-1) > np.finfo('float').eps:
        raise ValueError(
            'A must be a probability vector, sum(A)= %i !=1' % int(sum(A)))
    if any([a < 0 for a in A]):
        raise ValueError(
            'A must be a probability vector, all the elements of must be >0')
    if rate.ndim == 0:
        cpp = _cpp_hom_stat(A=A, t_stop=t_stop, r=rate, t_start=t_start)
    else:
        cpp = _cpp_het_stat(A=A, t_stop=t_stop, r=rate, t_start=t_start)
    if jitter is None:
        return cpp
    else:
        cpp = [
            dither_spike_train(cp, shift=jitter, edges=True)[0]
            for cp in cpp]
        return cpp
