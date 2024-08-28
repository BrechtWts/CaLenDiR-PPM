"""Contains functionality for validation set inference of the 
SEP-LSTM benchmark. NOTE that this is only for intermediate 
inference on the validation set, and not on the final test set. 

Following the original implementation of the SEP-LSTM ([1]_), validation 
set inference between epochs is only done for next activity and next 
timestamp prediction. 

References
----------
.. [1] N. Tax, I. Verenich, M. La Rosa, and M. Dumas, Predictive business 
       process monitoring with lstm neural networks, in Proc. of CAiSE, 
       LNCS, Springer, 2017
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from OneStepAheadBenchmarks.inference_utils_SEP import MultiOutputMetric

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from CaLenDiR_Utils.weighted_metrics_utils import get_weight_tensor, compute_corrected_avg



def inference_loop(model, 
                   inference_dataset, 
                   mean_std_ttne, 
                   num_classes, 
                   og_caseint,
                   val_batch_size):
    """Inference loop, both for validition set and ultimate test set.

    Parameters
    ----------
    model : SEP_Benchmarks_time
        Current version of the model. 
    inference_dataset : tuple of torch.Tensor
        Contains the tensors comprising the inference dataset, including 
        the labels for all prediction heads. 
    mean_std_ttne : list of float
        Training mean and standard deviation used to standardize the time 
        till next event (in seconds) target. Needed for re-converting 
        ttne predictions to original scale. Mean is the first entry, 
        std the second.
    num_classes : int 
        The number of distinct activity labels, including the padding 
        token (at index 0) and the END token (last index)
    og_caseint : torch.Tensor 
        Tensor of dtype torch.int64 and shape `(N_inf,)`, with `N_inf` 
        the number of instances contained within the inference (test or 
        validation) set. 
        Contains the integer-mapped case IDs of the 
        original inference set cases from which each of the `N_val` 
        instances have been derived. Used for computing the CaLenDiR 
        (weighted) metrics instead of the instance-based metrics. These 
        metrics are used for early stopping and final callback selection. 
    val_batch_size : int 
        Validation set batch size to loop over. 
    """

    inf_tensordataset = TensorDataset(*inference_dataset)
    inference_dataloader = DataLoader(inf_tensordataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=True)
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():

        # Initializing tensors storing the evaluation metrics for one 
        # step ahead prediction
        MAE_std_glob = torch.tensor(data=[], dtype=torch.float32).to(device)
        CE_inference_glob = torch.tensor(data=[], dtype=torch.float32).to(device)
        accuracy_glob = torch.tensor(data=[], dtype=torch.float32).to(device)

        # Composite loss metric used during training, used for LR scheduler benchmark 
        val_loss = torch.tensor(data=[], dtype=torch.float32).to(device)

        vloss_metric = MultiOutputMetric(num_classes)

        for valbatch_num, vdata in tqdm(enumerate(inference_dataloader), desc="Validation batch calculation"):
                vinputs = vdata[:-3]
                vlabels = vdata[-3:]
                vinputs = [vinput_tensor.clone().to(device) for vinput_tensor in vinputs]
                # Selecting only the TTNE and ACTIVITY LABELS 
                vlabels = [vlabels[0].to(device), vlabels[-1].to(device)]

                # Model predictions for activity and rrt suffix 
                voutputs = model(vinputs) 

                # Compute val loss and components 
                batch_losses = vloss_metric(voutputs, vlabels) 

                # Composite loss 
                vloss_batch = batch_losses[0] # (batch_size,)
                val_loss = torch.cat(tensors=(val_loss, vloss_batch), dim=-1) # (batch_size,)

                # Cross Entropy Loss component 
                CE_batch = batch_losses[1] # (batch_size,)
                CE_inference_glob = torch.cat(tensors=(CE_inference_glob, CE_batch), dim=-1)

                # MAE of the standardized TTNE targets
                MAE_std_batch = batch_losses[2] # (batch_size,)
                MAE_std_glob = torch.cat(tensors=(MAE_std_glob, MAE_std_batch), dim=-1) 

                # Accuracy boolean tensor next activity prediction
                acc_batch = batch_losses[-1] # (batch_size, )
                accuracy_glob = torch.cat(tensors=(accuracy_glob, acc_batch), dim=-1) 
                
        # Final validation metrics:
        # ------------------------- 

        #############################################
        # Instance-based (default) metric computation
        #############################################

        # Computing the number of instances (prefixes) in the inference set 
        num_prefs = val_loss.shape[0]

        # Average composite validation loss on the inference set  
        val_loss_avg = (torch.sum(val_loss) / num_prefs).item() 

        # Average Cross Entropy loss on the inference set 
        CE_avg = (torch.sum(CE_inference_glob) / num_prefs).item()

        # Average standardized MAE TTNE prediction
        avg_MAE_stand = (torch.sum(MAE_std_glob) / num_prefs).item()

        # Computing the average MAE in minutes 
        avg_MAE_min = (avg_MAE_stand*mean_std_ttne[1])/60

        # The next activity prediction accuracy 
        act_accuracy = (torch.sum(accuracy_glob.to(torch.float32)) / num_prefs).item()

        # Consolidating IB One-Step-Ahead Validation metrics 
        return_list_IB = [val_loss_avg, CE_avg, avg_MAE_stand, avg_MAE_min, act_accuracy]

        #############################################
        #       Case-based metric computation       #
        #############################################

        # Get weight tensor for each instance such that each original inference 
        # set case contributes equally, regardless of the original case's sequence 
        # length / total number of events. 
        # 
        # - weights : torch.Tensor of shape (num_prefs,) and dtype torch.float32 
        # - num_cases : integer, denoting the original number of cases in the inference 
        #               set from which the 'num_prefs' instances have been derived. 
        weights, num_cases = get_weight_tensor(og_caseint=og_caseint)
        weights = weights.to(device)

        # Case-Based (CB) Composite validation loss 
        val_loss_avg_CB = compute_corrected_avg(metric_tens=val_loss, 
                                                weight_tens=weights, 
                                                num_cases=num_cases)
        
        # CB average CE loss inference set 
        CE_avg_CB = compute_corrected_avg(metric_tens=CE_inference_glob, 
                                          weight_tens=weights, 
                                          num_cases=num_cases)
        
        # CB average standardized MAE TTNE prediction 
        avg_MAE_stand_CB = compute_corrected_avg(metric_tens=MAE_std_glob, 
                                                 weight_tens=weights, 
                                                 num_cases=num_cases)
        
        # CB average MAE TTNE in minutes 
        avg_MAE_min_CB = (avg_MAE_stand_CB*mean_std_ttne[1])

        # CB average next activity accuracy 
        act_accuracy_CB = compute_corrected_avg(metric_tens=accuracy_glob, 
                                                weight_tens=weights, 
                                                num_cases=num_cases)
        
        return_list_CB = [val_loss_avg_CB, CE_avg_CB, avg_MAE_stand_CB, avg_MAE_min_CB, act_accuracy_CB]


    return return_list_IB, return_list_CB