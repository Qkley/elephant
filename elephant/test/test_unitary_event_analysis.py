"""
Various functions to test the ue_utils package

@author: Rostami
"""

import unittest
import numpy as np
import quantities as pq
import elephant.unitary_event_analysis as ue


class UETestCase(unittest.TestCase):
    
    def test_hash_default(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        expected = np.array([77,43,23])
        h = ue.hash(m,N=8)
        self.assertTrue(np.all(expected == h))

    def test_hash_default_longpattern(self):
        m = np.zeros((100,2))
        m[0,0] = 1
        expected = np.array([2**99,0])
        h = ue.hash(m,N=100)
        self.assertTrue(np.all(expected == h))

    def test_hash_ValueError(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError,ue.hash,m,N=3)

    def test_hash_base_not_two(self):
        m = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        m = m.T
        base = 3
        expected = np.array([0,9,3,1,12,10,4,13])
        h = ue.hash(m,N=3,base=base)
        self.assertTrue(np.all(expected == h))

    ## TODO: write a test for ValueError in inv_hash
    def test_invhash_ValueError(self):
        self.assertRaises(ValueError,ue.inv_hash,[128,8],4)

    def test_invhash_default_base(self):
        N = 3
        h = np.array([0, 4, 2, 1, 6, 5, 3, 7])
        expected = np.array([[0, 1, 0, 0, 1, 1, 0, 1],[0, 0, 1, 0, 1, 0, 1, 1],[0, 0, 0, 1, 0, 1, 1, 1]])
        m = ue.inv_hash(h,N)
        self.assertTrue(np.all(expected == m))

    def test_invhash_base_not_two(self):
        N = 3
        h = np.array([1,4,13])
        base = 3
        expected = np.array([[0,0,1],[0,1,1],[1,1,1]])
        m = ue.inv_hash(h,N,base)
        self.assertTrue(np.all(expected == m))

    def test_invhash_shape_mat(self):
        N = 8
        h = np.array([178, 212, 232])
        expected = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
        m = ue.inv_hash(h,N)
        self.assertTrue(np.shape(m)[0] == N)

    def test_hash_invhash_consistency(self):
        m = np.array([[0, 0, 0],[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0],[1, 0, 1],[0, 1, 1],[1, 1, 1]])
        inv_h = ue.hash(m,N=8)
        m1 = ue.inv_hash(inv_h, N = 8)
        self.assertTrue(np.all(m == m1))

    def test_n_emp_mat_default(self):
        mat = np.array([[0, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 0, 1, 1, 1],[1, 0, 1, 1, 1]])
        N = 4
        pattern_hash = [3, 15]
        expected1 = np.array([ 2.,  1.])
        expected2 = [[0, 2], [4]]
        nemp,nemp_indices = ue.n_emp_mat(mat,N,pattern_hash)
        self.assertTrue(np.all(nemp == expected1))
        for item_cnt,item in enumerate(nemp_indices):
            self.assertTrue(np.allclose(expected2[item_cnt],item))

    def test_n_emp_mat_sum_trial_default(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([4,6])
        N = 3
        expected1 = np.array([ 1.,  3.])
        expected2 = [[[0], [3]],[[],[2,4]]]
        n_emp, n_emp_idx = ue.n_emp_mat_sum_trial(mat, N,pattern_hash)
        self.assertTrue(np.all(n_emp == expected1))
        for item0_cnt,item0 in enumerate(n_emp_idx):
            for item1_cnt,item1 in enumerate(item0):
                self.assertTrue(np.allclose(expected2[item0_cnt][item1_cnt],item1))

    def test_n_emp_mat_sum_trial_ValueError(self):
        mat = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError,ue.n_emp_mat_sum_trial,mat,N=2,pattern_hash = [3,6])

    def test_n_exp_mat_default(self):
        mat = np.array([[0, 0, 0, 1, 1],[0, 0, 0, 0, 1],[1, 0, 1, 1, 1],[1, 0, 1, 1, 1]])
        N = 4
        pattern_hash = [3, 11]
        expected = np.array([ 1.536,  1.024])
        nexp = ue.n_exp_mat(mat,N,pattern_hash)
        self.assertTrue(np.allclose(expected,nexp))

    def test_n_exp_mat_sum_trial_default(self):
        mat = np.array([[[1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 0, 1]],

                        [[1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1],
                         [1, 1, 0, 1, 0]]])
        pattern_hash = np.array([5,6])
        N = 3
        expected = np.array([ 1.56,  2.56])
        n_exp = ue.n_exp_mat_sum_trial(mat, N,pattern_hash)
        self.assertTrue(np.allclose(n_exp,expected))

    def test_n_exp_mat_sum_trial_ValueError(self):
        mat = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0],
                      [1,0,1],[0,1,1],[1,1,1]])
        self.assertRaises(ValueError,ue.n_exp_mat_sum_trial,mat,N=2,pattern_hash = [3,6])


def suite():
    suite = unittest.makeSuite(UETestCase, 'test')
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

