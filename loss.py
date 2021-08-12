import torch
import torchvision
import torch.nn as nn

class Focal_CE_weights(nn.Module):
  """
    This loss function is a variation of the cross entropy loss function, which adds class weights (calculated before hand) and
    the Focal Loss modification.
  """

  def __init__(self, **kwargs):
    super(Focal_CE_weights, self).__init__()
    self.kwargs = kwargs

  def forward(self, inputs, targets, smooth=1, gamma=2):
    """
      inputs: raw output probabilities*. *nn.CrossEntropyLoss already incorporates LogSoftmax()
      targets: one-hot encoded Ground Truth mask.

      gamma: focal loss downweighting parameter.
    """

    #FOCAL CROSS ENTROPY---
    _, targets = torch.max(targets, dim=1) #[b, 7, h, w] -> [b, 1, h, w] Undo one-hot encoding

    class_weights = torch.tensor([4.7e-3, 0.225, 0.264, 0.02, 0.348, 0.136]) #Calculated as the inverse number of samples and normalized
    class_weights = class_weights.to(device) #Weights to GPU

    BCE_loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)(inputs, targets)

    #Focal loss: easily classified pixels have a lesser contribution
    pt = torch.exp(-BCE_loss)
    F_loss = (1-pt)**gamma * BCE_loss

    return torch.mean(F_loss)
