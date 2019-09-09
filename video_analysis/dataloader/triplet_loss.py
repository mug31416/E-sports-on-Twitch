import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


# From
# https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
class BalancedBatchSampler(BatchSampler):
  """
  BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
  Returns batches of size n_classes * n_samples
  """

  def __init__(self, labels, n_classes, n_samples):
    self.labels = labels
    self.labels_set = list(set(self.labels))
    self.label_to_indices = {label: np.where(self.labels == label)[0]
                             for label in self.labels_set}
    for l in self.labels_set:
      np.random.shuffle(self.label_to_indices[l])
    self.used_label_indices_count = {label: 0 for label in self.labels_set}
    self.count = 0
    self.n_classes = n_classes
    self.n_samples = n_samples
    self.n_dataset = len(self.labels)
    self.batch_size = self.n_samples * self.n_classes

  def __iter__(self):
    self.count = 0
    while self.count + self.batch_size < self.n_dataset:
      classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
      indices = []
      for class_ in classes:
        indices.extend(self.label_to_indices[class_][
                       self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                               class_] + self.n_samples])
        self.used_label_indices_count[class_] += self.n_samples
        if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
          np.random.shuffle(self.label_to_indices[class_])
          self.used_label_indices_count[class_] = 0
      yield indices
      self.count += self.n_classes * self.n_samples

  def __len__(self):
    return self.n_dataset // self.batch_size


#FROM
# https://github.com/adambielski/siamese-triplet/blob/master/losses.py
class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
      super(OnlineTripletLoss, self).__init__()
      self.margin = margin
      self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
      triplets = self.triplet_selector.get_triplets(embeddings, target)

      if embeddings.is_cuda:
        triplets = triplets.cuda()

      ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
      an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
      losses = F.relu(ap_distances - an_distances + self.margin)

      return losses.mean(), len(triplets)


#FROM
# https://github.com/adambielski/siamese-triplet/blob/master/utils.py
def pdist(vectors):
  distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
    dim=1).view(-1, 1)
  return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
      pass

    def get_triplets(self, embeddings, labels):
      raise NotImplementedError


class AllTripletSelector(TripletSelector):
  """
  Returns all possible triplets
  May be impractical in most cases
  """

  def __init__(self):
    super(AllTripletSelector, self).__init__()

  def get_triplets(self, embeddings, labels):
    labels = labels.cpu().data.numpy()
    triplets = []
    for label in set(labels):
      label_mask = (labels == label)
      label_indices = np.where(label_mask)[0]
      if len(label_indices) < 2:
        continue
      negative_indices = np.where(np.logical_not(label_mask))[0]
      anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

      # Add all negatives for all positive pairs
      temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                       for neg_ind in negative_indices]
      triplets += temp_triplets

    return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
  hard_negative = np.argmax(loss_values)
  return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
  hard_negatives = np.where(loss_values > 0)[0]
  return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
  semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
  return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
  """
  For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
  Margin should match the margin used in triplet loss.
  negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
  and return a negative index for that pair
  """

  def __init__(self, margin, negative_selection_fn, cpu=True):
    super(FunctionNegativeTripletSelector, self).__init__()
    self.cpu = cpu
    self.margin = margin
    self.negative_selection_fn = negative_selection_fn

  def get_triplets(self, embeddings, labels):
    if self.cpu:
      embeddings = embeddings.cpu()
    distance_matrix = pdist(embeddings)
    distance_matrix = distance_matrix.cpu()

    labels = labels.cpu().data.numpy()
    triplets = []

    for label in set(labels):
      label_mask = (labels == label)
      label_indices = np.where(label_mask)[0]
      if len(label_indices) < 2:
        continue
      negative_indices = np.where(np.logical_not(label_mask))[0]
      anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
      anchor_positives = np.array(anchor_positives)

      ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
      for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
        loss_values = ap_distance - distance_matrix[
          torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
        loss_values = loss_values.data.cpu().numpy()
        hard_negative = self.negative_selection_fn(loss_values)
        if hard_negative is not None:
          hard_negative = negative_indices[hard_negative]
          triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

    if len(triplets) == 0:
      triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

    triplets = np.array(triplets)

    return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                               negative_selection_fn=lambda
                                                                                                 x: semihard_negative(x,
                                                                                                                      margin),
                                                                                               cpu=cpu)

#FROM
# https://github.com/adambielski/siamese-triplet/blob/master/Experiments_MNIST.ipynb
def extract_embeddings(dataloader, model):
  with torch.no_grad():
    model.eval()
    embeddings = np.zeros((len(dataloader.dataset), 2))
    labels = np.zeros(len(dataloader.dataset))
    k = 0
    for images, target in dataloader:
      if cuda:
        images = images.cuda()
      embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
      labels[k:k + len(images)] = target.numpy()
      k += len(images)
  return embeddings, labels

