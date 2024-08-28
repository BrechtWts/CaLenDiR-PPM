"""
Module contains funtionality for computing the suffix-length normalized 
validation loss metric for the CRTP-LSTM benchmark model. 


Following the authors of the original implementation ([1]_), early 
stopping as well as a Learning Rate Scheduler are implemented based on 
consecutive non-improvements in the 
validation loss function, which is the same metric used for training. 

To ensure a fair and controlled setup, a CaLenDiR version of this 
validation loss, in which Suffix-Length Normalization is applied 
on the loss suffixes, is implemented. As such, a scalar average 
is obtained per instance for both the activity suffix (categorical 
crossentropy) and remaining runtime suffix (Mean Absolute Error - MAE). 

By applying Uniform CaseBased Weighting over the N averages, with N being 
the number of instances / prefix-suffix pairs in the validation set, 
each original case assigned to the validation set, contributes equally 
to the validation loss. 



References
----------
.. [1] B. R. Gunnarsson, S. v. Broucke and J. De Weerdt, "A 
        Direct Data Aware LSTM Neural Network Architecture for 
        Complete Remaining Trace and Runtime Prediction," in IEEE 
        Transactions on Services Computing, vol. 16, no. 4, pp. 
        2330-2342, 1 July-Aug. 2023, doi: 10.1109/TSC.2023.3245726.
"""
import torch 
import torch.nn as nn

class SuffixLengthNormalizedCrossEntropyMetric(nn.Module):
    def __init__(self, num_classes):
        super(SuffixLengthNormalizedCrossEntropyMetric, self).__init__()
        # Number of activity output neurons. Includes padding token and end_token.
        self.num_classes = num_classes
        # Padding token at index 0
        self.cross_entropy_crit = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        
    def forward(self, inputs, targets):
        """Compute the Suffix-Length Categorical Cross-Entropy (CCE) Loss 
        for activity suffix prediction. The loss is normalized based on 
        the lengths of the suffixes, ensuring equal contributions from 
        sequences (suffixes) of varying lengths, mitigating a bias in the 
        batch losses towards longer cases. This is achieved by averaging 
        the Cross-Entropy values for the whole suffix per instance first, 
        masking out the padding tokens as well. 
        Thereby, an average per instance is obtained, and returned. 

        Afterwards (not implemented in this function), Uniform Case-Based-
        Weighting is applied to get the CaLenDiR version of the CCE 
        validation loss, in which each case gets an equal weight in the 
        final scalar CCE metric. Consequently, the case length 
        distribution of the original cases assigned to the validation 
        (or test) set, and used for deriving the validation (or test) set 
        instances, is reflected in the computation of the validation 
        loss. 

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
        normalized_loss : torch.Tensor
            Contains the average (suffix-length-normalized) masked CCE 
            validation loss for each of the batch_size instances, for 
            remaining runtime suffix prediction (CRTP-LSTM). Shape 
            (batch_size,) and dtype torch.float32
        """
        # Derive batch and window size
        batch_size = inputs.shape[0]
        window_size = inputs.shape[1]

        # Reshape to (batch_size*window_size, num_classes)
        inputs = torch.reshape(inputs, shape=(-1, self.num_classes))

        # Reshape to (batch_size*window_size, )
        targets = torch.reshape(targets, shape=(-1,))

        # Compute masked CE for all activity suffix predictions
        loss = self.cross_entropy_crit(inputs, targets) # shape (batch_size*window_size,)

        # Reshape back to (batch_size, window_size)
        loss = torch.reshape(loss, (batch_size, window_size))



        # Compute actual suffix length each instance 
        suf_len = torch.sum((torch.reshape(targets, (batch_size, window_size))!=0).to(torch.float32), dim=-1) # (batch_size,)

        # Compute average loss over each instance (suffix) - suffix length normalization
        normalized_loss = loss.sum(dim=1) / suf_len # (batch_size,)

        return normalized_loss 
    

class SuffixLengthNormalizedMAEMetric(nn.Module):
    def __init__(self):
        super(SuffixLengthNormalizedMAEMetric, self).__init__()
        
    def forward(self, inputs, targets):
        """Compute the Suffix-Length Normalized Mean Absolute Error (MAE) 
        loss for timestamp suffix prediction. The loss is normalized 
        based on the lengths of the suffixes, ensuring equal 
        contributions from sequences (suffixes) of varying lengths, 
        mitigating a bias in the batch losses towards longer cases. This 
        is achieved by averaging the MAE values for the whole suffix 
        per instance first, masking out the padding tokens as well. 
        Thereby, an average per instance is obtained, after which the 
        final loss is computed by taking the mean over all instances.
        
        Masking is achieved by masking out target values of -100.0, which 
        correspond to padded event tokens.

        Parameters
        ----------
        inputs : torch.Tensor
            The tensor containing the continuous predictions for the 
            timestamp suffix target. Shape (batch_size, window_size, 1) 
            and dtype torch.float32. For the CRTP-LSTM model, this tensor 
            contains the remaining runtime suffix predictions. 
        targets : torch.Tensor
            The continuous time prediction targets. Shape 
            (batch_size, window_size, 1), dtype torch.float32. 

        Returns
        -------
        avg_instance_mae : torch.Tensor
            The average (suffix-length-normalized) masked MAE loss for 
            each of the batch_size validation instances, for 
            remaining runtime suffix prediction (CRTP-LSTM). 
            Scalar tensor of shape (batch_size,) and of dtype 
            torch.float32. 
        """
        # Discard trailing dimension of size 1 
        inputs = inputs[:,:,0] # (batch_size, window_size)
        targets = targets[:, :, 0] # (batch_size, window_size)

        # Create mask to ignore time targets with value -100
        mask = (targets != -100).float() # (batch_size, window_size)

        # Compute absolute errors 
        absolute_erros = torch.abs(inputs-targets) # (batch_size, window_size)

        # Replace MAE values pertaining to padding events with 0 
        masked_absolute_erros = absolute_erros * mask # (batch_size, window_size)

        # Compute average loss over each instance (suffix) - suffix length normalization
        avg_instance_mae = torch.sum(masked_absolute_erros, dim=-1) / torch.sum(mask, dim=-1) # (batch_size, )


        return avg_instance_mae
    

class MaskedSuffixLengthNormalizedMultiOutputMetric(nn.Module):
    def __init__(self, num_classes):
        """Composite Suffix-Length Normalized Validation Loss for the 
        following two jointly learned prediction tasks (CRTP-LSTM): 

        #. activity suffix prediction 

        #. remaining runtime suffix prediction
        
        The loss is normalized based on 
        the lengths of the suffixes, ensuring equal contributions from 
        sequences (suffixes) of varying lengths, mitigating a bias in the 
        batch losses towards longer cases. This is achieved by averaging 
        the suffix loss values for the whole suffix per instance first, 
        masking out the padding tokens as well. 
        Thereby, an average per instance is obtained, and returned. 

        Afterwards (not implemented in this function), Uniform Case-Based-
        Weighting is applied to get the CaLenDiR version of the 
        validation loss, in which each case gets an equal weight in the 
        final scalar metric. Consequently, the case length 
        distribution of the original cases assigned to the validation 
        (or test) set, and used for deriving the validation (or test) set 
        instances, is reflected in the computation of the validation 
        loss. 

        Parameters
        ----------
        num_classes : int
            Number of output neurons (including padding and end tokens) 
            in the output layer of the activity suffix prediction task. 
        """
        super(MaskedSuffixLengthNormalizedMultiOutputMetric, self).__init__()
        self.cat_loss_fn = SuffixLengthNormalizedCrossEntropyMetric(num_classes)
        self.cont_loss_fn_ttne = SuffixLengthNormalizedMAEMetric()

    def forward(self, outputs, labels):
        """Compute composite loss components for validation inference. 
        Needed for the learning rate scheduler used in the CRTP_LSTM 
        benchmark. Accounts for masking, subsetting, for each of the 
        `batch_size` instances, only the loss contributions pertaining 
        to actual non-padded events. 

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
            Tensor of shape (batch_size,), containing the average 
            validation loss for each instance, averaged over the (ground-
            truth) suffix lenghts. 
        """
        # Loss activity suffix prediction
        cat_loss = self.cat_loss_fn(outputs[0], labels[-1]) # shape (batch_size,)
        
        # Remaining runtime (rrt) suffix prediction
        cont_loss = self.cont_loss_fn_ttne(outputs[1], labels[-2]) # shape (batch_size,)

        # Composite loss (used for gradient updates)
        loss = cat_loss + cont_loss # shape (batch_size,)

        return loss # shape (NP,)