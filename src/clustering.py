import math
import numpy as np
import random
import torch
from scipy.spatial.distance import cdist

class Fast_KMeans:
  def __init__(self, n_clusters, centroids = None, max_iter=100, tol=0.0001, verbose=0, mode="euclidean", minibatch=None):
    """
        Initialize a Fast_KMeans instance.

        Args:
            n_clusters (int): The number of clusters.
            centroids (numpy.ndarray, optional): Initial cluster centroids. Default is None.
            max_iter (int, optional): Maximum number of iterations. Default is 100.
            tol (float, optional): Tolerance for convergence. Default is 0.0001.
            verbose (int, optional): Verbosity level. Default is 0.
            mode (str, optional): Distance metric mode, 'euclidean' or 'cosine'. Default is 'euclidean'.
            minibatch (int, optional): Size of minibatch for minibatch K-means. Default is None.

      """
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.mode = mode
    self.minibatch = minibatch
    self._loop = False
    self._show = False

    try:
      import PYNVML
      self._pynvml_exist = True
    except ModuleNotFoundError:
      self._pynvml_exist = False
    
    self.centroids = centroids

  @staticmethod
  def cos_sim(a, b):
    a_norm = a.norm(dim=-1, keepdim=True)
    b_norm = b.norm(dim=-1, keepdim=True)
    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)
    return a @ b.transpose(-2, -1)

  @staticmethod
  def euc_sim(a, b):
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

  def remaining_memory(self):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if self._pynvml_exist:
      pynvml.nvmlInit()
      gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
      remaining = info.free
    else:
      remaining = torch.cuda.memory_allocated()
    return remaining

  def max_sim(self, a, b):
    device = a.device.type
    batch_size = a.shape[0]
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim

    if device == 'cpu':
      if b.device.type != 'cpu':
        b = b.cpu()
      sim = sim_func(a, b)
      max_sim_v, max_sim_i = sim.max(dim=-1)
      return max_sim_v, max_sim_i
    else:
      if a.dtype == torch.float:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
      elif a.dtype == torch.half:
        expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
      ratio = math.ceil(expected / self.remaining_memory())
      subbatch_size = math.ceil(batch_size / ratio)
      msv, msi = [], []
      for i in range(ratio):
        if i*subbatch_size >= batch_size:
          continue
        sub_x = a[i*subbatch_size: (i+1)*subbatch_size]
        sub_sim = sim_func(sub_x, b)
        sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
        del sub_sim
        msv.append(sub_max_sim_v)
        msi.append(sub_max_sim_i)
      if ratio == 1:
        max_sim_v, max_sim_i = msv[0], msi[0]
      else:
        max_sim_v = torch.cat(msv, dim=0)
        max_sim_i = torch.cat(msi, dim=0)
      return max_sim_v, max_sim_i

  def fit_predict(self, X, centroids=None):
    """
    Fit the K-means model to the data and predict cluster labels for input data.

    Args:
        X (numpy.ndarray): Input data.
        centroids (numpy.ndarray, optional): Initial cluster centroids. Default is None.

    Returns:
        numpy.ndarray: Cluster labels for input data.
    """

    batch_size, emb_dim = X.shape
    device = X.device.type
    # start_time = time()
    if centroids is None:
      self.centroids = X[np.random.choice(batch_size, size=[self.n_clusters], replace=False)]
    else:
      self.centroids = centroids
    num_points_in_clusters = torch.ones(self.n_clusters, device=device)
    closest = None
    for i in range(self.max_iter):
      # iter_time = time()
      if self.minibatch is not None:
        x = X[np.random.choice(batch_size, size=[self.minibatch], replace=False)]
      else:
        x = X
      closest = self.max_sim(a=x, b=self.centroids)[1]
      matched_clusters, counts = closest.unique(return_counts=True)

      c_grad = torch.zeros_like(self.centroids)
      if self._loop:
        for j, count in zip(matched_clusters, counts):
          c_grad[j] = x[closest==j].sum(dim=0) / count
      else:
        if self.minibatch is None:
          expanded_closest = closest[None].expand(self.n_clusters, -1)
          mask = (expanded_closest==torch.arange(self.n_clusters, device=device)[:, None]).float()
          c_grad = mask @ x / mask.sum(-1)[..., :, None]
          c_grad[c_grad!=c_grad] = 0 # remove NaNs
        else:
          expanded_closest = closest[None].expand(len(matched_clusters), -1)
          mask = (expanded_closest==matched_clusters[:, None]).float()
          c_grad[matched_clusters] = mask @ x / mask.sum(-1)[..., :, None]


      error = (c_grad - self.centroids).pow(2).sum()
      if self.minibatch is not None:
        lr = 1/num_points_in_clusters[:,None] * 0.9 + 0.1
      else:
        lr = 1
      num_points_in_clusters[matched_clusters] += counts
      self.centroids = self.centroids * (1-lr) + c_grad * lr
      if error <= self.tol:
        break

    return closest

  def predict(self, X):
    """
    Predict cluster labels for input data.

    Args:
        X (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Cluster labels for input data.
    """
    return self.max_sim(a=X, b=self.centroids)[1]

  def fit(self, X, centroids=None):
    """
    Fit the K-means model to the data.

    Args:
        X (numpy.ndarray): Input data.
        centroids (numpy.ndarray, optional): Initial cluster centroids. Default is None.
    """
    self.fit_predict(X, centroids)


def init_centers(trainx, k):
  """
  Initialize cluster centers for K-means clustering.

  Args:
      trainx (numpy.ndarray): Training data.
      k (int): Number of clusters.

  Returns:
      numpy.ndarray: Cluster centers.
      list: Indices of selected cluster centers.
  """
  c = 0
  n_classes = len(trainx)
  counter = 0
  while(counter < 1000):
      a = random.sample(range(n_classes), k=k)
      d = np.sum(cdist(trainx[a], trainx[a], 'cosine') > 0.69) * 1
      if d == (len(a) ** 2 - len(a)):
          break
      print(c, d, (len(a) ** 2 - len(a)))
      c += 1
      counter += 1
  return trainx[a], a