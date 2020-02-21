import numpy as np
from functools import reduce

class Tiling:
  def __init__(self, X, n_tilings):
    self.tiling = np.array([np.linspace(x[0], x[1], x[2] + 1, endpoint=True) for x in X])
    self.n_tilings = n_tilings
    self.dim = len(X)
    odd_numbers = 2*np.arange(self.dim) + 1
    spacing = self.tiling[:,1] - self.tiling[:,0]
    self.offset = odd_numbers * spacing / n_tilings

  def reducer(self, a, b):
        return np.dot(a.reshape(-1,1), b.reshape(1,-1)).reshape(-1,)

  def pt_to_feat(self, x):
    a = self.tiling[:,:-1] <= x.reshape(-1,1)
    b = self.tiling[:,1:] > x.reshape(-1,1)
    c = a & b
    d = reduce(self.reducer, c, np.array([1]))
    return d

  def pt_to_feat_offset(self, x, k):
    x = x + k*self.offset
    return self.pt_to_feat(x)

  def pt_to_feats(self, x):
    feats = np.array([self.pt_to_feat_offset(x, k) for k in np.arange(self.n_tilings) - np.floor(self.n_tilings/2)])
    return feats.reshape(-1,)