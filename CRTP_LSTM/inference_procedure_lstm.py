"""Functionality for conducting parallel inference over batches for the 
CRTP-LSTM benchmark model. Returning all metrics both aggregated 
over all prefix lengths, as well as individually for each prefix length.

The CRTP-LSTM, unlike SuTraN and ED-LSTM, does not generate suffixes in 
an autoregressive (AR) manner. 
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from CRTP_LSTM.inference_environment_lstm import BatchInference
from CRTP_LSTM.inference_utils_lstm import MultiOutputMetric
from CRTP_LSTM.inference_utils_masked import MaskedMultiOutputMetric

from torch.utils.data import TensorDataset, DataLoader
import os 
import pickle

# Importing functionality for computing Uniform Case-Based Weighted metrics 
from CaLenDiR_Utils.weighted_metrics_utils import get_weight_tensor, compute_corrected_avg, suflen_normalized_ttne_mae

# Importing functionality for computing Suffix-Length Normalized vaidation loss CRTP-LSTM
from CaLenDiR_Utils.CRTP_LSTM_calendir_utils import MaskedSuffixLengthNormalizedMultiOutputMetric

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_loop(model, 
                   inference_dataset, 
                   num_categoricals_pref,
                   mean_std_ttne, 
                   mean_std_tsp, 
                   mean_std_tss, 
                   mean_std_rrt, 
                   og_caseint, 
                #    masking=True, 
                   results_path=None, 
                   val_batch_size=8192):
    """Inference loop, both for validition set and ultimate test set.

    Parameters
    ----------
    model : CRTP_LSTM
        The initialized and current version of a CRTP-LSTM neural 
        network. Should be in evaluation mode already. 
    inference_dataset : tuple of torch.Tensor
        Contains the tensors comprising the inference dataset, including 
        the labels for all prediction tasks. 
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token. 
    mean_std_ttne : list of float
        Training mean and standard deviation used to standardize the time 
        till next event (in seconds) target. Needed for re-converting 
        ttne predictions to original scale. Mean is the first entry, 
        std the second.
    mean_std_tsp : list of float
        Training mean and standard deviation used to standardize the time 
        since previous event (in seconds) feature of the decoder suffix 
        tokens. Needed for re-converting time since previous event values 
        to original scale (seconds). Mean is the first entry, std the 2nd.
    mean_std_tss : list of float
        Training mean and standard deviation used to standardize the time 
        since start (in seconds) feature of the decoder suffix tokens. 
        Needed for re-converting time since start to original scale 
        (seconds). Mean is the first entry, std the 2nd. 
    mean_std_rrt : list of float, optional
        List consisting of two floats, the training mean and standard 
        deviation of the remaining runtime labels (in seconds). Needed 
        for de-standardizing remaining runtime predictions and labels, 
        such that the MAE can be expressed in seconds (and minutes). 
    og_caseint : torch.Tensor 
        Tensor of dtype torch.int64 and shape `(N_inf,)`, with `N_inf` 
        the number of instances contained within the inference (test or 
        validation) set. 
        Contains the integer-mapped case IDs of the 
        original inference set cases from which each of the `N_val` 
        instances have been derived. Used for computing the CaLenDiR 
        (weighted) metrics instead of the instance-based metrics. These 
        metrics are used for early stopping and final callback selection. 
    masking : bool, optional 
        Whether or not the (right-padded) padding tokens for the activity 
        suffix and remaining runtime suffix labels should be padded 
        during training, and hence whether the validation metric should 
        account for the paddings too. `True` by default. 
    results_path : None or str, optional
        The absolute path name of the folder in which the final evaluation results 
        should be stored. The default of None should be retained for 
        intermediate validation set computations.
    val_batch_size : int, optional
        Batch size for iterating over inference dataset. By default 8192. 
    """

    # Creating TensorDataset and corresponding DataLoader out of 
    # `inference_dataset`. 
    inf_tensordataset = TensorDataset(*inference_dataset)
    inference_dataloader = DataLoader(inf_tensordataset, batch_size=val_batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # Retrieving labels 
    labels_global = inference_dataset[-3:] 

    # Retrieving seq length (`window_size`, also referred to as W) 
    act_label_index = -1
    window_size = labels_global[act_label_index].shape[-1]
    num_prefs = labels_global[act_label_index].shape[0]
    num_classes = torch.max(labels_global[act_label_index]).item() + 1
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # Initializing global tensors for storing model outputs on CPU
        # The two directly underneath have shape 
        # (number of instances, window_size) after inference loop
        suffix_acts_decoded_global = torch.empty((0, window_size), dtype=torch.int64)
        suffix_rrt_preds_global = torch.empty((0, window_size), dtype=torch.float32)

        # Evaluation RRT metrics: only the first rrt prediction is to be used upon 
        # evaluation
        MAE_rrt_stand_global = torch.tensor(data=[], dtype=torch.float32).to(device)
        MAE_rrt_seconds_global = torch.tensor(data=[], dtype=torch.float32).to(device)

        # MAE_ttne_seconds_global = torch.tensor(data=[], dtype=torch.float32).to(device)
        MAE_ttne_seconds_global = torch.empty((0, window_size), dtype=torch.float32).to(device)


        # # Informational metrics regarding the differenes between length 
        # # predicted activity suffix vs length ground truth suffix 
        # Omitted for this paper. In case one wants to keep track of these 
        # metrics too, uncomment the line of code underneath. 

        # length_diff_global = torch.tensor(data=[], dtype=torch.int64).to(device)
        # length_diff_too_early_global = torch.tensor(data=[], dtype=torch.int64).to(device)
        # length_diff_too_late_global = torch.tensor(data=[], dtype=torch.int64).to(device)
        # amount_right_global = torch.tensor(data=0, dtype=torch.int64).to(device) # scalar tensor

        # Initializing a global tensor to store the prefix lengths of all inference instances 
        pref_len_global = torch.tensor(data=[], dtype=torch.int64).to(device)

        # Initializing a global tensor to store the suffix lengths of all inference instances 
        suf_len_global = torch.tensor(data=[], dtype=torch.int64).to(device)

        # normalized Damerau Levenshtein similarity metric
        dam_lev_global = torch.tensor(data=[], dtype=torch.int64).to(device)


        # if masking:
        # Only with masking from now on - empirically confirmed to deliver 
        # more competitive results. 
        # Default composite validation loss metric used during training, 
        # i.e. used for LR scheduler and early stopping CRTP-LSTM 
        # benchmark in case of default training.
        vloss_metric = MaskedMultiOutputMetric(num_classes)
        
        # Case-Based (CB) vloss metric in case of CaLenDiR training, used 
        # for LR scheduler and early stopping CRTP-LSTM benchmark in case 
        # of default training
        vloss_metric_CB = MaskedSuffixLengthNormalizedMultiOutputMetric(num_classes)
        # else:
        #     vloss_metric = MultiOutputMetric(num_classes)

        # Initialize global tensor for storing default composite 
        # validation loss metric values. 
        val_loss = torch.tensor(data=[], dtype=torch.float32).to(device)

        # Initialize global tensor for storing Case-Based (CB) composite 
        # validation loss metric values. 
        val_loss_CB = torch.tensor(data=[], dtype=torch.float32).to(device)
        

        for valbatch_num, vdata in tqdm(enumerate(inference_dataloader), desc="Validation batch calculation"):
                vinputs = vdata[:-3]
                vlabels = vdata[-3:]

                # Prefix padding mask (right padded): 
                pad_mask = vdata[num_categoricals_pref+1]
                pad_mask = pad_mask.to(device) # (batch_size, window_size)

                # Sending inputs and labels to GPU
                vinputs = [vinput_tensor.clone().to(device) for vinput_tensor in vinputs]
                vlabels = [vlabel_tensor.clone().to(device) for vlabel_tensor in vlabels]

                act_labels = vlabels[-1]

                # Deriving suffix length of each instance 
                suf_len = torch.argmax((act_labels == (num_classes-1)).to(torch.int64), dim=-1) + 1 # (batch_size,)
                suf_len_global = torch.cat(tensors=(suf_len_global, suf_len), dim=-1)

                # Deriving the prefix length of each instance 
                padding_idx = torch.argmax(pad_mask.to(torch.int64), dim=-1) # (batch_size,)
                pref_len_global = torch.cat(tensors=(pref_len_global, padding_idx), dim=-1)

                # Model predictions for activity and rrt suffix 
                voutputs = model(vinputs) 

                # Compute val loss (default)
                vloss_batch = vloss_metric(voutputs, vlabels) # (batch_size*window_size,)
                val_loss = torch.cat(tensors=(val_loss, vloss_batch), dim=-1) # (batch_size,)

                # Compute CB val loss 
                vloss_batch_CB = vloss_metric_CB(voutputs, vlabels) # (batch_size,)
                val_loss_CB = torch.cat((val_loss_CB, vloss_batch_CB), dim=-1)



                # Initialize inference environment
                # This immediately generates a variety of auxiliary 
                # tensors needed for computing the evaluation metrics. 
                infer_env = BatchInference(preds=voutputs,
                                           labels=vlabels,
                                           mean_std_ttne=mean_std_ttne, 
                                           mean_std_tsp=mean_std_tsp, 
                                           mean_std_tss=mean_std_tss, 
                                           mean_std_rrt=mean_std_rrt)
                # Retrieving the predicted activity and ttne suffixes 
                suffix_acts_decoded = infer_env.suffix_acts_decoded.clone() # (batch_size, window_size)
                suffix_acts_decoded_global = torch.cat((suffix_acts_decoded_global, suffix_acts_decoded.cpu()), dim=0) 

                suffix_rrt_preds = infer_env.rrt_preds.clone() # (batch_size, window_size)
                suffix_rrt_preds_global = torch.cat((suffix_rrt_preds_global, suffix_rrt_preds.cpu()), dim=0) 

                # # Computing all inference metrics 
                MAE_ttne_seconds = infer_env.compute_ttne_results() # shape (batch_size, window_size)
                MAE_ttne_seconds_global = torch.cat(tensors=(MAE_ttne_seconds_global, MAE_ttne_seconds), dim=0)


                # (normalized) Damerau-Levenshtein distance activity suffix prediction
                dam_lev = infer_env.damerau_levenshtein_distance_tensors() # (batch_size, )
                dam_lev_global = torch.cat(tensors=(dam_lev_global, dam_lev), dim=-1)

                # MAE remainining runtime predictions (rrt)
                # Evaluation RRT metrics: only the first rrt prediction is to be used upon 
                # evaluation
                MAE_rrt_stand, MAE_rrt_seconds = infer_env.compute_rrt_results()
                MAE_rrt_stand_global = torch.cat(tensors=(MAE_rrt_stand_global, MAE_rrt_stand), dim=-1) # (batch_size,)
                MAE_rrt_seconds_global = torch.cat(tensors=(MAE_rrt_seconds_global, MAE_rrt_seconds), dim=-1) # (batch_size,)
                
                # Length differences between predicted and ground-truth suffixes. 
                # Omitted for this paper. In case one wants to keep track of these 
                # metrics too, uncomment the line of code underneath. 
                # length_diff, length_diff_too_early, length_diff_too_late, amount_right = infer_env.compute_suf_length_diffs()
                # length_diff_global = torch.cat(tensors=(length_diff_global, length_diff), dim = -1)
                # length_diff_too_early_global = torch.cat(tensors=(length_diff_too_early_global, length_diff_too_early), dim=-1)
                # length_diff_too_late_global = torch.cat(tensors=(length_diff_too_late_global, length_diff_too_late), dim=-1)
                # amount_right_global += amount_right

        # Correction pref len 
        # replacing 0s with max_pref_len 
        pref_len_global = torch.where(pref_len_global == 0, window_size, pref_len_global) # (num_prefs,)
        # Write away results for final test set inference if specified 
        if results_path: 
            subfolder_path = results_path
            os.makedirs(subfolder_path, exist_ok=True)

            # Specifying paths to save the prediction tensors and writing 
            # them to disk. 
            #   Activity suffix predictions 
            suffix_acts_decoded_path = os.path.join(subfolder_path, 'suffix_acts_decoded.pt')
            torch.save(suffix_acts_decoded_global, suffix_acts_decoded_path)

            #   Remaining runtime (rrt) suffix predictions 
            suffix_rrt_preds_path = os.path.join(subfolder_path, 'suffix_rrt_preds.pt')
            torch.save(suffix_rrt_preds_global, suffix_rrt_preds_path)
            
            # Prefix length and suffix length 
            pref_len_path = os.path.join(subfolder_path, 'pref_len.pt')
            torch.save(pref_len_global.cpu(), pref_len_path)

            suf_len_path = os.path.join(subfolder_path, 'suf_len.pt')
            torch.save(suf_len_global.cpu(), suf_len_path)

            # Labels 
            # Note, this is a tuple of tensors instead of just a tensor
            labels_path = os.path.join(subfolder_path, 'labels.pt')
            torch.save(labels_global, labels_path)

        # Final validation metrics:
        # ------------------------- 

        dam_lev_similarity = 1. - dam_lev_global # shape (num_prefs, )

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

        # Activity suffix 
        # Case-Based (CB) average DLS 
        avg_dam_lev_CB = compute_corrected_avg(metric_tens=dam_lev_similarity, 
                                               weight_tens=weights, 
                                               num_cases=num_cases)
        
        # Remaining Runtime MAE 
        # CB average MAE (both standardized, in seconds and in minutes)
        #   Standardized CB
        avg_MAE_stand_RRT_CB = compute_corrected_avg(metric_tens=MAE_rrt_stand_global, 
                                                        weight_tens=weights, 
                                                        num_cases=num_cases)
        #   Seconds 
        avg_MAE_seconds_RRT_CB = compute_corrected_avg(metric_tens=MAE_rrt_seconds_global, 
                                                        weight_tens=weights, 
                                                        num_cases=num_cases)
        #   Minutes 
        avg_MAE_minutes_RRT_CB = avg_MAE_seconds_RRT_CB / 60 


        # Timestamp suffix prediction 
        #   Computing average per instance (Suffix-Length Normalization)
        MAE_ttne_seconds_CB = suflen_normalized_ttne_mae(MAE_ttne=MAE_ttne_seconds_global.cpu().clone(), 
                                                         suf_len=suf_len_global.cpu()) # (num_prefs,)
        
        MAE_ttne_minutes_CB = MAE_ttne_seconds_CB / 60 # (num_prefs,)

        #   CB averages TTNE prediction

        #       seconds 
        avg_MAE_ttne_seconds_CB = compute_corrected_avg(metric_tens=MAE_ttne_seconds_CB, 
                                                        weight_tens=weights.cpu(), 
                                                        num_cases=num_cases) # scalar 
        
        #       minutes 
        avg_MAE_ttne_minutes_CB = avg_MAE_ttne_seconds_CB / 60 # scalar 


        # CaLenDiR composite validation loss CRTP-LSTM
        # (Case-Based Weighting of the `num_prefs` Suffix-Length-
        # Normalized composite losses)
        avg_val_loss_CB = compute_corrected_avg(metric_tens=val_loss_CB, 
                                                weight_tens=weights, 
                                                num_cases=num_cases) # scalar 


        #############################################
        # Instance-based (default) metric computation
        #############################################

        # Time Till Next Event (TTNE) suffix 
        #     Retain only MAE contributions pertaining to 
        #     non-padded suffix events
        counting_tensor = torch.arange(window_size, dtype=torch.int64).to(device) # (window_size,)
        #       Repeat the tensor along the first dimension to match the desired shape
        counting_tensor = counting_tensor.unsqueeze(0).repeat(num_prefs, 1) # (num_prefs, window_size)
        #       Compute boolean indexing tensor to, for each of the 
        #       'num_prefs' instances, slice out only the absolute 
        #       errors pertaining to actual non-padded suffix events. 
        before_end_token = counting_tensor <= (suf_len_global-1).unsqueeze(-1) # (num_prefs,)

        avg_MAE_ttne_seconds = MAE_ttne_seconds_global[before_end_token] # shape (torch.sum(suf_len_global), )
        avg_MAE_ttne_seconds = (torch.sum(avg_MAE_ttne_seconds) / avg_MAE_ttne_seconds.shape[0]).item()

        avg_MAE_ttne_minutes = avg_MAE_ttne_seconds / 60

        #   normalized Damerau Levenshtein similarity Activity Suffix prediction 
        avg_dam_lev = (torch.sum(dam_lev_global) / dam_lev_global.shape[0]).item()
        avg_dam_lev = 1 - avg_dam_lev # scalar 
        
        # Without averaging 
        # dam_lev_similarity = 1. - dam_lev_global # (num_prefs,)

        #   Evaluation RRT metrics: only based on first remaining runtime predictions
        avg_MAE_stand_RRT = (torch.sum(MAE_rrt_stand_global) / MAE_rrt_stand_global.shape[0]).item()
        avg_MAE_seconds_RRT = (torch.sum(MAE_rrt_seconds_global) / MAE_rrt_seconds_global.shape[0]).item()
        avg_MAE_minutes_RRT = avg_MAE_seconds_RRT / 60

        # Without averaging
        MAE_rrt_minutes = MAE_rrt_seconds_global / 60 # (num_prefs, )

        if results_path:
            # Writing the tensors containing the DLS and MAE RRT for each individual 
            # test set instance / test set prefix-suffix pair, to disk. 
            dam_lev_sim_path = os.path.join(subfolder_path, 'dam_lev_similarity.pt')
            torch.save(dam_lev_similarity, dam_lev_sim_path)

            MAE_rrt_minutes_path = os.path.join(subfolder_path, 'MAE_rrt_minutes.pt')
            torch.save(MAE_rrt_minutes, MAE_rrt_minutes_path)

        # # Evaluation metrics ttne:
        # avg_MAE_ttne_seconds = (torch.sum(MAE_ttne_seconds_global) / MAE_ttne_seconds_global.shape[0]).item()
        # avg_MAE_ttne_minutes = avg_MAE_ttne_seconds / 60
        
        # Length differences: 
        # Omitted for this paper. In case one wants to keep track of these 
        # metrics too, uncomment the line of code underneath. 
        # total_num = length_diff_global.shape[0]
        # num_too_early = length_diff_too_early_global.shape[0]
        # num_too_late = length_diff_too_late_global.shape[0]
        # percentage_too_early = num_too_early / total_num
        # percentage_too_late = num_too_late / total_num
        # percentage_correct = amount_right_global.item() / total_num
        # mean_absolute_length_diff = (torch.sum(torch.abs(length_diff_global)) / total_num).item()
        # mean_too_early = (torch.sum(torch.abs(length_diff_too_early_global)) / num_too_early).item()
        # mean_too_late = (torch.sum(torch.abs(length_diff_too_late_global)) / num_too_late).item()

        # Computing average validation loss (default)
        val_loss_avg = (torch.sum(val_loss) / val_loss.shape[0]).item()

    ################################################
    # Consolidating instance-based (IB) (default) metrics 
    ################################################

    # return_list = [avg_dam_lev, percentage_too_early, percentage_too_late]
    # return_list += [percentage_correct, mean_absolute_length_diff, mean_too_early, mean_too_late]

    # return_list += [avg_MAE_stand_RRT, avg_MAE_minutes_RRT, val_loss_avg, avg_MAE_ttne_minutes]

    return_list_IB = [avg_dam_lev, avg_MAE_stand_RRT, avg_MAE_minutes_RRT, val_loss_avg, avg_MAE_ttne_minutes]

    ################################################
    # Consolidating Case-Based metrics 
    ################################################

    return_list_CB = [avg_dam_lev_CB, avg_MAE_stand_RRT_CB, avg_MAE_minutes_RRT_CB, avg_val_loss_CB, avg_MAE_ttne_minutes_CB]


    ##################################################
    #   Computing average metrics for instances of   #
    #      different prefix and suffix lengths       #
    ##################################################

    # Making dictionaries of the results for over both prefix and suff length. 
    results_dict_pref = {}
    for i in range(1, window_size+1):
        bool_idx = pref_len_global==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        MAE_ttne_i = MAE_ttne_minutes_CB[bool_idx.cpu()].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            avg_mae_ttne = (torch.sum(MAE_ttne_i) / num_inst).item()
            results_i = [avg_dl, avg_mae, avg_mae_ttne, num_inst]
            results_dict_pref[i] = results_i
    results_dict_suf = {}
    for i in range(1, window_size+1):
        bool_idx = suf_len_global==i
        dam_levs = dam_lev_similarity[bool_idx].clone()
        MAE_rrt_i = MAE_rrt_minutes[bool_idx].clone()
        MAE_ttne_i = MAE_ttne_minutes_CB[bool_idx.cpu()].clone()
        num_inst = dam_levs.shape[0]
        if num_inst > 0:
            avg_dl = (torch.sum(dam_levs) / num_inst).item()
            avg_mae = (torch.sum(MAE_rrt_i) / num_inst).item()
            avg_mae_ttne = (torch.sum(MAE_ttne_i) / num_inst).item()
            results_i = [avg_dl, avg_mae, avg_mae_ttne, num_inst]
            results_dict_suf[i] = results_i
    
    pref_suf_results = [results_dict_pref, results_dict_suf]


    if results_path:
        # Writing the prefix and suffix dictionaries to disk 

        #   Prefix length results dictionary 
        preflen_results_path = os.path.join(subfolder_path, 'prefix_length_results_dict.pkl')
        with open(preflen_results_path, 'wb') as file:
            pickle.dump(results_dict_pref, file)

        #   Suffix length results dictionary 
        suflen_results_path = os.path.join(subfolder_path, 'suffix_length_results_dict.pkl')
        with open(suflen_results_path, 'wb') as file:
            pickle.dump(results_dict_suf, file)

    return return_list_IB, return_list_CB, pref_suf_results