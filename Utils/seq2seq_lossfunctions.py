"""
Contains the two individual loss functions for activity suffix and time 
suffix prediction used for instance-based (default) training by all 
seq2seq models (SuTraN, ED-LSTM and CRTP-LSTM). 
"""

import torch
import torch.nn as nn

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(MaskedCrossEntropyLoss, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        # Padding token at index 0
        self.cross_entropy_crit = nn.CrossEntropyLoss(ignore_index = 0)
        
    def forward(self, inputs, targets):
        """Compute the CrossEntropyLoss of the activity suffix prediction 
        head while masking the predictions coresponding to padding events. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the unnormalized logits for each 
            activity class. Shape (batch_size, window_size, num_classes) 
            and dtype torch.float32.
        targets : torch.Tensor
            The activity labels, containing the indices. Shape 
            (batch_size, window_size), dtype torch.int64. 

        Returns
        -------
        loss: torch.Tensor
            The masked cross entropy loss for the activity prediction head. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Reshape inputs to shape (batch_size*window_size, num_classes)
        inputs = torch.reshape(input=inputs, shape=(-1, self.num_classes))
        # Reshape targets to shape (batch_size*window_size,)
        targets = torch.reshape(input=targets, shape=(-1,))

        # Compute masked loss 
        loss = self.cross_entropy_crit(inputs, targets) # scalar tensor

        return loss
    
class MaskedMeanAbsoluteErrorLoss(nn.Module):
    def __init__(self):
        super(MaskedMeanAbsoluteErrorLoss, self).__init__()
        
    def forward(self, inputs, targets):
        """Computes the Mean Absolute Error (MAE) loss in which the 
        target values of -100.0, corresponding to padded event tokens, 
        are ignored / masked and hence do not contribute to the input 
        gradient. 

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for the 
            timestamp suffix target. Shape (batch_size, window_size, 1) 
            and dtype torch.float32. For the CRTP-LSTM model, this tensor 
            contains the remaining runtime suffix predictions. 
        targets : torch.Tensor
            The time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. 

        Returns
        -------
        loss: torch.Tensor
            The masked MAE loss for one of the time prediction heads. 
            Scalar tensor (shape (,)) of dtype torch.float32. 
        """
        # Reshape inputs to shape (batch_size*window_size,)
        inputs = torch.reshape(input=inputs, shape=(-1,))
        # Reshape targets to shape (batch_size*window_size,)
        targets= torch.reshape(input=targets, shape=(-1,))

        # Create mask to ignore time targets with value -100
        mask = (targets != -100).float()

        absolute_errors = torch.abs(inputs-targets) # (batch_size * window_size,)

        masked_absolute_errors = absolute_errors * mask # (batch_size * window_size,)

        # count: number of non-ignored targets 
        count = torch.sum(mask)

        # Compute masked loss 
        return torch.sum(masked_absolute_errors) / count 