# -*- coding: utf-8 -*-
'''
Function to calculate spike-triggered averages of analog signals.

:copyright: Copyright 2015 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
'''

import numpy as np
import quantities as pq
from neo.core import AnalogSignalArray, SpikeTrain


def spike_triggered_average(signal, spiketrains, window):
    """
    Calculates the spike-triggered average of analog signals in a given time window around 
    the spike times of a corresponding spiketrain for multiple signals each.

    The function receives n analog signals and either one or n spiketrains. In case it is one spiketrain 
    this one is muliplied n-fold and used for each of the n analog signals.

    Parameters
    ----------

        signal: neo AnalogSignalArray object containing n analog signals
        spiketrains: One SpikeTrain or one numpy ndarray or a list of n of either of those
        window: pair (2-tuple) of a starttime and a stoptime (relative to a spike) 
                of the time interval for signal averaging

    Returns
    -------

        Returns an AnalogSignalArray of spike-triggered averages of the respective spiketrains.
        The length of the respective array is calculated as the number of window_bins from the 
        given start- and stoptime of the averaging interval and the sampling rate of the analog signal.
        If for an analog signal no spike was either given or all given spikes had to be ignored because 
        of a too large analysis window, the corresponding analog signal is returned as nan.

    Example
    -------

        result = spike_triggered_average(signal, [spiketrain1, spiketrain2], (-5 * ms, 10 * ms))

    """

    # checking compatibility of data and data types
    # window_starttime: time to specify the starttime of the averaging interval relative to a spike
    # window_stoptime: time to specify the stoptime of the averaging interval relative to a spike
    window_starttime, window_stoptime = window
    if not (isinstance(window_starttime, pq.quantity.Quantity) and window_starttime.dimensionality.simplified == pq.Quantity(1, "s").dimensionality):
        raise TypeError("window_starttime must be a time quantity.")
    if not (isinstance(window_stoptime, pq.quantity.Quantity) and window_stoptime.dimensionality.simplified == pq.Quantity(1, "s").dimensionality):
        raise TypeError("window_stoptime must be a time quantity.")
    if window_stoptime <= window_starttime:
        raise ValueError(
            "window_starttime must be earlier than window_stoptime.")

    # checks on signal
    if not isinstance(signal, AnalogSignalArray):
        raise TypeError(
            "Signal must be an AnalogSignalArray, not %s." % type(signal))
    if len(signal.shape) > 1:
        # num_signals: number of analog signals
        num_signals = signal.shape[1]
    else:
        raise ValueError("Empty analog signal, hence no averaging possible.")
    if window_stoptime - window_starttime > signal.t_stop - signal.t_start:
        raise ValueError(
            "The chosen time window is larger than the time duration of the signal.")

    # spiketrains type check
    if isinstance(spiketrains, (np.ndarray, SpikeTrain)):
        spiketrains = [spiketrains]
    elif isinstance(spiketrains, list):
        for st in spiketrains:
            if not isinstance(st, (np.ndarray, SpikeTrain)):
                raise TypeError(
                    "spiketrains must be a SpikeTrain, a numpy ndarray, or a list of one of those, not %s." % type(spiketrains))
    else:
        raise TypeError(
            "spiketrains must be a SpikeTrain, a numpy ndarray, or a list of one of those, not %s." % type(spiketrains))

    # multiplying spiketrain in case only a single spiketrain is given
    if len(spiketrains) == 1 and num_signals != 1:
        template = spiketrains[0]
        spiketrains = []
        for i in range(num_signals):
            spiketrains.append(template)

    # checking for matching numbers of signals and spiketrains
    if num_signals != len(spiketrains):
        raise ValueError(
            "The number of signals and spiketrains has to be the same.")

    # checking the times of signal and spiketrains
    for i in range(num_signals):
        if spiketrains[i].t_start < signal.t_start:
            raise ValueError(
                "The spiketrain indexed by %i starts earlier than the analog signal." % i)
        if spiketrains[i].t_stop > signal.t_stop:
            raise ValueError(
                "The spiketrain indexed by %i stops later than the analog signal." % i)

    # *** Main algorithm: ***

    # window_bins: number of bins of the chosen averaging interval
    window_bins = int(np.round(
        ((window_stoptime - window_starttime) * signal.sampling_rate).simplified))
    # result_sta: array containing finally the spike-triggered averaged signal
    result_sta = AnalogSignalArray(np.zeros(
        (window_bins, num_signals)), sampling_rate=signal.sampling_rate, units=signal.units)
    # setting of correct times of the spike-triggered average relative to the spike
    result_sta.t_start = window_starttime
    used_spikes = np.zeros(num_signals, dtype=int)
    unused_spikes = np.zeros(num_signals, dtype=int)
    total_used_spikes = 0

    for i in range(num_signals):
        # summing over all respective signal intervals around spiketimes
        for spiketime in spiketrains[i]:
            # checks for sufficient signal data around spiketime
            if spiketime + window_starttime >= signal.t_start and spiketime + window_stoptime <= signal.t_stop:
                # calculating the startbin in the analog signal of the averaging window for spike
                startbin = int(np.round(
                    ((spiketime + window_starttime - signal.t_start) * signal.sampling_rate).simplified))
                # adds the signal in selected interval relative to the spike
                result_sta[:, i] += signal[startbin: startbin + window_bins, i]
                # counting of the used spikes
                used_spikes[i] += 1
            else:
                # counting of the unused spikes
                unused_spikes[i] += 1

        # normalization
        result_sta[:, i] = result_sta[:, i] / used_spikes[i]

        total_used_spikes += used_spikes[i]

    if total_used_spikes == 0:
        raise ValueError(
            "No spike at all was either found or used for averaging")
    result_sta.annotate(used_spikes=used_spikes, unused_spikes=unused_spikes)

    return result_sta
