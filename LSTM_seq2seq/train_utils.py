"""
Custom masked loss functions for the ED-LSTM benchmark implementation. 
"""

import torch
import torch.nn as nn

#####################################
##    Individual loss functions    ##
#####################################

from Utils.seq2seq_lossfunctions import MaskedCrossEntropyLoss, MaskedMeanAbsoluteErrorLoss
from CaLenDiR_Utils.seq2seq_norm_lossfunctions import SuffixLengthNormalizedCrossEntropyLoss, SuffixLengthNormalizedMAELoss

class MultiOutputLoss(nn.Module):
    def __init__(self, num_classes, clen_dis_ref):
        """Composite loss function for the following two jointly 
        learned prediction tasks: 

        #. activity suffix prediction (default)

        #. time till next event suffix predicion (default)
        
        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        clen_dis_ref : bool 
            If `True`, Case Length Distribution-Reflective (CaLenDiR) 
            Training is performed, and hence Suffix-Length-Normalized 
            Loss Functions are used for training. If `False`, the default 
            training procedure, in which no loss function normalization is 
            performed (and hence in which case-length distortion is not 
            addressed), is used. 
        """
        super(MultiOutputLoss, self).__init__()
        if clen_dis_ref:
            self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = SuffixLengthNormalizedMAELoss()
        else:
            self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_ttne = MaskedMeanAbsoluteErrorLoss()

    # def forward(self, cat_output, ttne_output, rrt_output, cat_target, ttne_target, rrt_target):
    def forward(self, outputs, labels):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress.

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of two tensors, each containing the 
            model's predictions for one of the two tasks. 
        labels : tuple of torch.Tensor
            Tuple consisting of two tensors, each containing the 
            labels for one of the two tasks.

        Returns
        -------
        loss : torch.Tensor
            Scalar tensor. Contains the composite loss that is used for 
            updating the gradients during training. Gradient tracking 
            turned on.
        cat_loss.item() : float
            Native python float. The (masked) cross entropy loss for 
            the next activity prediction head. Not used for gradient 
            updates during training, but for keeping track of the 
            different loss components during training and evaluation.
        cont_loss.item() : float
            Native python float. The (masked) MAE loss for the time 
            till next event prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1])
        
        # Loss Time Till Next Event (ttne) suffix prediction
        cont_loss = self.cont_loss_fn_ttne(outputs[1], labels[0])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss

        # Composite loss, act suffix loss, ttne loss
        return loss, cat_loss.item(), cont_loss.item()