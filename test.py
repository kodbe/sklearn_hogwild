import unittest
from hogwildsgd import HogWildRegressor
import scipy.sparse
import numpy as np
import pdb


class TestHogwild(unittest.TestCase):

    def test_work(self):
        X = scipy.sparse.random(20000,10, density=.2).toarray() # Guarantees sparse grad updates
        real_w = np.random.uniform(0,1,size=(10,1))
        y = np.dot(X,real_w)

        #pdb.set_trace()
        hw = HogWildRegressor(n_jobs = 4, 
                              n_epochs = 5,
                              batch_size = 1, 
                              chunk_size = 32,
                              learning_rate = .001,
                              generator=None,
                              verbose=2)
        #pdb.set_trace()
        hw = hw.fit(X,y)

        #print('predictions: ', hw.predictions)
        y = y.reshape((len(y),))
        for pred in hw.predictions:
            score = np.mean(abs(y-pred))
            print('score: ', score)


        #y_hat = hw.predict(X)
        #print('yhat: ', y_hat.shape)
        #y = y.reshape((len(y),))
        #print('y: ', y.shape)
        #score = np.mean(abs(y-y_hat))
        #print('score: ', score)
        #self.assertTrue(score < .005) 


if __name__ == '__main__':
    unittest.main()