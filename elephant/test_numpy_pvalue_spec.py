import sys
import unittest

import neo
import numpy as np
import quantities as pq

import spike_train_generation as gen
import spade


class PVTestCase(unittest.TestCase):

    def test_pvalue_spec_2d(self):
        rate = 40 * pq.Hz
        refr_period = 4 * pq.ms
        t_start = 0. * pq.ms
        t_stop = 1000. * pq.ms
        num_spiketrains = 20

        binsize = 3 * pq.ms
        winlen = 5
        dither = 10 * pq.ms
        n_surr = 10
        min_spikes = 2
        min_occ = 2
        max_spikes = 10
        max_occ = None
        min_neu = 2
        spectrum = '#'

        np.random.seed(0)
        hpr = gen.homogeneous_poisson_process_with_refr_period
        sts = [hpr(rate, refr_period, t_start, t_stop)
               for ind in range(num_spiketrains)]

        np.random.seed(0)
        print()
        print('Old p-value spec', spectrum)
        pv_spec = spade.pvalue_spectrum(
            sts, binsize, winlen, dither=dither,
            n_surr=n_surr, min_spikes=min_spikes,
            min_occ=min_occ, max_spikes=max_spikes,
            max_occ=max_occ, min_neu=min_neu,
            spectrum=spectrum)

        np.random.seed(0)
        print('New p-value spec', spectrum)
        pv_spec_np = spade.pvalue_spectrum_numpy(
            sts, binsize, winlen, dither=dither,
            n_surr=n_surr, min_spikes=min_spikes,
            min_occ=min_occ, max_spikes=max_spikes,
            max_occ=max_occ, min_neu=min_neu,
            spectrum=spectrum)

        self.assertIsInstance(pv_spec_np, list)
        self.assertEqual(len(pv_spec_np), len(pv_spec))
        for entry_np in pv_spec_np:
            self.assertIsInstance(entry_np, list)
            for entry_id, entry in enumerate(pv_spec):
                if entry_np[0] == entry[0] and entry_np[1] == entry[1]:
                    self.assertAlmostEqual(entry_np[2], entry[2])
                    pv_spec.pop(entry_id)
                break
            else:
                raise AssertionError('This entry {} was not found'.format(
                    entry_np))

    def test_pvalue_spec_3d(self):
        rate = 40 * pq.Hz
        refr_period = 4 * pq.ms
        t_start = 0. * pq.ms
        t_stop = 1000. * pq.ms
        num_spiketrains = 20

        binsize = 3 * pq.ms
        winlen = 5
        dither = 10 * pq.ms
        n_surr = 10
        min_spikes = 2
        min_occ = 2
        max_spikes = 10
        max_occ = None
        min_neu = 2
        spectrum = '3d#'

        np.random.seed(0)
        hpr = gen.homogeneous_poisson_process_with_refr_period
        sts = [hpr(rate, refr_period, t_start, t_stop)
               for ind in range(num_spiketrains)]

        np.random.seed(0)
        print()
        print('Old p-value spec', spectrum)
        pv_spec = spade.pvalue_spectrum(
            sts, binsize, winlen, dither=dither,
            n_surr=n_surr, min_spikes=min_spikes,
            min_occ=min_occ, max_spikes=max_spikes,
            max_occ=max_occ, min_neu=min_neu,
            spectrum=spectrum)

        np.random.seed(0)
        print('New p-value spec', spectrum)
        pv_spec_np = spade.pvalue_spectrum_numpy(
            sts, binsize, winlen, dither=dither,
            n_surr=n_surr, min_spikes=min_spikes,
            min_occ=min_occ, max_spikes=max_spikes,
            max_occ=max_occ, min_neu=min_neu,
            spectrum=spectrum)

        self.assertIsInstance(pv_spec_np, list)
        self.assertEqual(len(pv_spec_np), len(pv_spec))
        for entry_np in pv_spec_np:
            self.assertIsInstance(entry_np, list)
            for entry_id, entry in enumerate(pv_spec):
                if entry_np[0] == entry[0] and entry_np[1] == entry[1] and\
                        entry_np[2] == entry[2]:
                    self.assertAlmostEqual(entry_np[3], entry[3])
                    pv_spec.pop(entry_id)
                break
            else:
                raise AssertionError('This entry {} was not found'.format(
                    entry_np))


def suite():
    suite = unittest.makeSuite(PVTestCase, 'test')
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
