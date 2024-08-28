"""
Custom masked loss functions for the CRTP-LSTM benchmark. In contrast to 
the loss function contained within `CRTP_LSTM\train_utils_lstm.py` 
in which the padded events are not masked and hence do contribute to the 
computation of losses, masking is applied here. 

The original implementation of the CRTP-LSTM benchmark did not include 
masking. The authors 
however indicated that after further exploration, it became evident that 
masking lead to faster convergence. Therefore, in the SuTraN paper, 
masking is applied for the CRTP-LSTM benchmark as well. 
"""

import torch
import torch.nn as nn


#####################################
##    Individual loss functions    ##
#####################################

from Utils.seq2seq_lossfunctions import MaskedCrossEntropyLoss, MaskedMeanAbsoluteErrorLoss
from CaLenDiR_Utils.seq2seq_norm_lossfunctions import SuffixLengthNormalizedCrossEntropyLoss, SuffixLengthNormalizedMAELoss

class MaskedMultiOutputLoss(nn.Module):
    def __init__(self, num_classes, clen_dis_ref):
        """Composite loss function for the following two jointly 
        learned prediction tasks: 

        #. activity suffix prediction 

        #. remaining runtime suffix predicion (default)
        
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
        super(MaskedMultiOutputLoss, self).__init__()
        if clen_dis_ref:
            self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_rrt = SuffixLengthNormalizedMAELoss()
        else:
            self.cat_loss_fn = MaskedCrossEntropyLoss(num_classes)
            self.cont_loss_fn_rrt = MaskedMeanAbsoluteErrorLoss()

    def forward(self, outputs, labels):
        """Compute composite loss (for gradient updates) and return its 
        components as python floats for tracking training progress of 
        the individual prediction heads as well. 

        Parameters
        ----------
        outputs : tuple of torch.Tensor
            Tuple consisting of two tensors, each containing the 
            model's predictions for one of the two tasks. 
        labels : tuple of torch.Tensor
            Tuple consisting of three tensors, each containing the 
            labels for one of the three tasks. However, since CRTP-LSTM 
            is trained on activity suffix and remaining runtime suffix 
            prediction only, only two out of three are needed. The last 
            tensor contains the activity suffix labels, the penultimate 
            tensor the remaining runtime suffix labels. 

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
            Native python float. The (masked) MAE loss for the remaining  
            runtime suffix prediction head. Not (directly) used for 
            gradient updates during training, but for keeping track of 
            the different loss components during training and evaluation.
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1])
        
        # Remaining runtime (rrt) suffix prediction
        cont_loss = self.cont_loss_fn_rrt(outputs[1], labels[-2])

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss

        return loss, cat_loss.item(), cont_loss.item()