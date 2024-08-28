import torch
import torch.nn as nn

from SuTraN.train_utils import MultiOutputLoss
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from SuTraN.inference_procedure import inference_loop

# Importing functionality for Uniform Case-Based Sampling (UCBS) 
# (part of CaLenDiR training)
from CaLenDiR_Utils.case_based_sampling import sample_train_instances, precompute_indices

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, 
                training_loader, 
                remaining_runtime_head, 
                outcome_bool,
                optimizer,
                loss_fn, 
                batch_interval,
                epoch_number, 
                max_norm):

    # Tracking global loss over all prediction heads:
    running_loss_glb = []
    # Tracking loss of each prediction head separately: 
    running_loss_act = [] # Cross-Entropy
    running_loss_ttne = [] # MAE

    # Creating auxiliary bools 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head
    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is remaining time prediction
    if only_rrt:
        running_loss_rrt = [] # MAE
        num_target_tens = 3
    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is (binary) outcome prediction
    elif only_out:
        running_loss_out = []
        num_target_tens = 3
    # If the additional prediction heads (on top of activity and ttne 
    # suffix) comprise both rrt and outcome prediction
    elif both:
        running_loss_rrt = []
        running_loss_out = []
        num_target_tens = 4
    # If their are no additional prediction heads (on top of activity and 
    # ttne suffix)
    elif both_not: 
        num_target_tens = 2

    original_norm_glb = []
    clipped_norm_glb = []



    for batch_num, data in tqdm(enumerate(training_loader), desc="Batch calculation at epoch {}.".format(epoch_number)):

        inputs = data[:-num_target_tens]
        labels = data[-num_target_tens:]
        # Sending inputs and labels to GPU
        inputs = [input_tensor.to(device) for input_tensor in inputs]
        labels = [label_tensor.to(device) for label_tensor in labels]

        # Restoring gradients to 0 for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss 
        loss_results = loss_fn(outputs, labels)
        loss = loss_results[0]

        # Compute gradients 
        loss.backward()

        # Keep track of original gradient norm 
        original_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        original_norm_glb.append(original_norm.item())

        # Clip gradient norm
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        clipped_norm_glb.append(clipped_norm.item())

        # Adjust learning weights
        optimizer.step()

        # Tracking losses and metrics
        running_loss_glb.append(loss.item())

        running_loss_act.append(loss_results[1])
        running_loss_ttne.append(loss_results[2])

        if only_rrt:
            running_loss_rrt.append(loss_results[-1])

        elif only_out: 
            running_loss_out.append(loss_results[-1])
        
        elif both:
            running_loss_rrt.append(loss_results[-2])
            running_loss_out.append(loss_results[-1])


        
        if batch_num % batch_interval == (batch_interval-1):
                print("------------------------------------------------------------")
                print("Epoch {}, batch {}:".format(epoch_number, batch_num))
                print("Average original gradient norm: {} (over last {} batches)".format(sum(original_norm_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Average clipped gradient norm: {} (over last {} batches)".format(sum(clipped_norm_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Running average global loss: {} (over last {} batches)".format(sum(running_loss_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Running average activity prediction loss: {} (Cross Entropy over last {} batches)".format(sum(running_loss_act[-batch_interval:])/batch_interval, batch_interval))
                print("Running average time till next event prediction loss: {} (MAE over last {} batches)".format(sum(running_loss_ttne[-batch_interval:])/batch_interval, batch_interval))
                if remaining_runtime_head:
                    print("Running average (complete) remaining runtime prediction loss: {} (MAE over last {} batches)".format(sum(running_loss_rrt[-batch_interval:])/batch_interval, batch_interval))
                if outcome_bool:
                    print("Running average outcome prediction loss: {} (BCE over last {} batches)".format(sum(running_loss_out[-batch_interval:])/batch_interval, batch_interval))
                print("------------------------------------------------------------")

    print("=======================================")
    print("End of epoch {}".format(epoch_number))
    print("=======================================")
    last_running_avg_glob = sum(running_loss_glb[-batch_interval:])/batch_interval
    print("Running average global loss: {} (over last {} batches)".format(last_running_avg_glob, batch_interval))
    last_running_avg_act = sum(running_loss_act[-batch_interval:])/batch_interval
    print("Running average activity prediction loss: {} (Cross Entropy over last {} batches)".format(last_running_avg_act, batch_interval))
    last_running_avg_ttne = sum(running_loss_ttne[-batch_interval:])/batch_interval
    print("Running average time till next event prediction loss: {} (MAE over last {} batches)".format(last_running_avg_ttne, batch_interval))

    if only_rrt:
        last_running_avg_rrt = sum(running_loss_rrt[-batch_interval:])/batch_interval
        print("Running average (complete) remaining runtime prediction loss: {} (MAE over last {} batches)".format(last_running_avg_rrt, batch_interval))
        last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, last_running_avg_rrt, loss
        return model, optimizer, last_running_avgs
    elif only_out:
        last_running_avg_out = sum(running_loss_out[-batch_interval:])/batch_interval
        print("Running average outcome prediction loss: {} (BCE over last {} batches)".format(last_running_avg_out, batch_interval))
        last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, last_running_avg_out, loss
        return model, optimizer, last_running_avgs
    elif both:
        last_running_avg_rrt = sum(running_loss_rrt[-batch_interval:])/batch_interval
        print("Running average (complete) remaining runtime prediction loss: {} (MAE over last {} batches)".format(last_running_avg_rrt, batch_interval))
        last_running_avg_out = sum(running_loss_out[-batch_interval:])/batch_interval
        print("Running average outcome prediction loss: {} (BCE over last {} batches)".format(last_running_avg_out, batch_interval))
        last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, last_running_avg_rrt, last_running_avg_out, loss
        return model, optimizer, last_running_avgs
    elif both_not:
        last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, loss
        return model, optimizer, last_running_avgs # model remains the same
            

def train_model(model, 
                optimizer, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                remaining_runtime_head, 
                outcome_bool, 
                num_classes, 
                batch_interval, 
                path_name, 
                num_categoricals_pref, 
                mean_std_ttne, 
                mean_std_tsp, 
                mean_std_tss, 
                mean_std_rrt, 
                batch_size, 
                clen_dis_ref, 
                og_caseint_train, 
                og_caseint_val,
                median_caselen,
                patience = 24, 
                lr_scheduler_present=False, 
                lr_scheduler=None, 
                best_MAE_ttne = 1e9, 
                best_DL_sim = -1, 
                best_MAE_rrt = 1e9, 
                best_BCE = 1e9, 
                best_auc_pr = -1,
                max_norm = 2.):
    """Outer training loop SuTraN. 

    Parameters
    ----------
    model : SuTraN or SuTraN_no_context
        The initialized and current version of a SuTraN neural network. 
        Should be set to evaluation mode to trigger the AR decoding loop 
        within SuTraN's forward method. 
    optimizer : torch optimizer
        torch.optim.AdamW optimizer. Should already be initialized and 
        wrapped around the parameters of `model`. 
    train_dataset : tuple of torch.Tensor
        Tuple containing the tensors comprising the training set. This 
        includes, i.a., the labels. All tensors have an outermost 
        dimension of the same size, i.e. `N_train`, the number of 
        original training set instances / prefix-suffix pairs. 
    val_dataset : tuple of torch.Tensor 
        Tuple containing the tensors comprising the validation set. This 
        includes, i.a., the labels. All tensors have an outermost 
        dimension of the same size, i.e. `N_val`, the number of 
        original validation set instances / prefix-suffix pairs. 
    start_epoch : int
        Number of the epoch from which the training loop is started. 
        First call to ``train_model()`` should be done with 
        ``start_epoch=0``.
    num_epochs : int
        Number of epochs to train. When resuming training with another 
        loop of num_epochs, for the new ``train_model()``, the new 
        ``start_epoch`` argument should be equal to the current one 
        plus the current value for ``num_epochs``.
    remaining_runtime_head : bool
        Whether or not the model is also trained to directly predict the 
        remaining runtime given a prefix. See Notes for further remarks 
        regarding the `remaining_runtime_head` parameter. 
    outcome_bool : bool, optional 
        Whether or not the model should also include a prediction 
        head for binary outcome prediction. If 
        `outcome_bool=True`, a prediction head for predicting 
        the binary outcome given a prefix is added. This prediction 
        head, in contrast to the time till next event and activity 
        suffix predictions, will only be trained to provide a 
        prediction at the first decoding step. Note that the 
        value of `outcome_bool` should be aligned with the 
        `outcome_bool` parameter of the model and train 
        procedure, as well as with the preprocessing pipeline that 
        produces the labels. See Notes for further remarks 
        regarding the `outcome_bool` parameter. 
    num_classes : int
        The number of output neurons for the activity prediction head. 
        This includes the padding token (0) and the END token. 
    batch_interval : int
        The periodic amount of batches trained for which the moving average 
        losses and metrics are printed and recorded. E.g. if 
        ``batch_interval=100``, then after every 100 batches, the 
        moving averages of all metrics and losses during training are 
        recorded, printed and reset to 0. 
    path_name : str 
        Needed for saving results and callbacks in the 
        appropriate subfolders. This is the path name 
        of the subfolder for which all the results 
        and callbacks (model copies) should be 
        stored for the current event log and 
        model configuration.
    num_categoricals_pref : int
        The number of categorical features (including the activity label) 
        contained within each prefix event token. Given the NDA 
        characteristic of ED-LSTM, this should be specified as one. 
        Allowed for flexibility for future studies. 
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
    batch_size : int 
        Batch size used during training. 
    clen_dis_ref : bool 
        If `True`, Case Length Distribution-Reflective (CaLenDiR) 
        Training is performed. This includes the application of Uniform 
        Case-Based Sampling (UCBS) of instances each epoch, and 
        Suffix-Length-Normalized Loss Functions. If `False`, the default 
        training procedure, in which all instances are used for training 
        each epoch and in which no loss function normalization is 
        performed (and hence in which case-length distortion is not 
        addressed), is performed. 
    og_caseint_train : torch.Tensor 
        Tensor of dtype torch.int64 and shape 
        `(N_train,)`. Contains the integer-mapped case IDs of the 
        original training set cases from which each of the `N_train` 
        instances have been derived. Used for Uniform Case-Based Sampling 
        (UCBS) in case CaLenDiR training is adopted. 
    og_caseint_val : torch.Tensor 
        Tensor of dtype torch.int64 and shape 
        `(N_val,)`. Contains the integer-mapped case IDs of the 
        original validation set cases from which each of the `N_val` 
        instances have been derived. Used for computing the CaLenDiR 
        (weighted) metrics instead of the instance-based metrics if 
        `clen_dis_ref=True`. These metrics are used for early stopping 
        and final callback selection. 
    median_caselen : int
        Median case length original cases. 
    patience : int, optional. 
        Max number of epochs without any improvement in any of the 
        validation metrics. After `patience` epochs without any 
        improvement, the training loop is terminated early. By default 24.
    lr_scheduler_present : bool, optional
        Indicates whether we work with a learning rate scheduler wrapped 
        around the optimizer. If True, learning rate scheduler 
        included. If False (default), not. 
    lr_scheduler : torch lr_scheduler or None
        If ``lr_scheduler_present=True``, a lr_scheduler that is wrapped 
        around the optimizer should be provided as well. For SuTraN, 
        the ExponentialLR() scheduler should be used. 
    best_MAE_ttne : float 
        Best validation Mean Absolute Error for the time till 
        next event suffix prediction. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. 
    best_DL_sim : float 
        Best validation 1-'normalized Damerau-Levenshtein distance for 
        activity suffix prediction so far. The defaults apply if the  
        training loop is initialized for the first time for a given 
        configuration. If the training loop is resumed from a certain 
        checkpoint, the best results of the previous training loop should 
        be given. 
    best_MAE_rrt : float
        Best validation Mean Absolute Error for the remaining runtime 
        prediction so far. The defaults apply if the training 
        loop is initialized for the first time for a given configuration. 
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. If 
        `remaining_runtime_head=False`, the defaults can be retained even 
        when resuming the training loop from a checkpoint. 
    best_BCE : float
        Best validation Binary Cross Entropy binary outcome prediction so 
        far. The defaults apply if the training loop is initialized for 
        the first time for a given configuration. If the training loop is 
        resumed from a certain checkpoint, the best results of the 
        previous training loop should be given. If 
        `outcome_bool=False`, the defaults can be retained even when 
        resuming the training loop from a checkpoint. 
    best_auc_pr : float
        Best validation score for area under the Precision Recall curve  
        so far (outcome prediction). The defaults apply if the training 
        loop is initialized for the first time for a given configuration.  
        If the training loop is resumed from a certain checkpoint, the 
        best results of the previous training loop should be given. If 
        `outcome_bool=False`, the defaults can be retained even when 
        resuming the training loop from a checkpoint. 
    max_norm : float, optional
        Max gradient norm used for clipping during training. By default 2.

    Notes
    -----
    Additional remarks regarding parameters: 

    * `remaining_runtime_head` : This parameter has become redundant, and 
      should always be set to `True`. SuTraN by default accounts for an 
      additional direct remaining runtime prediction head. 

    * `outcome_bool` : For the paper implementation, this boolean should 
      be set to `False`. For future work, already included for extending 
      the multi-task PPM setup to simultaneously predict a binary outcome 
      target for each prefix as well.  
    """
    if lr_scheduler_present:
        if lr_scheduler==None:
            print("No lr_scheduler provided.")
            return -1, -1, -1, -1

    # Checking whether GPU is being used
    print("Device: {}".format(device))

    # Assigning model to GPU. 
    model.to(device)

    if clen_dis_ref:
        print("CaLenDiR training activated")
    else:
        print("Default training mode. CaLenDiR training not activated.")
    
    # in case of default training, DataLoader can be specified over all 
    # instances once. 
    if not clen_dis_ref:
        train_tens_dataset = TensorDataset(*train_dataset)
        train_dataloader = DataLoader(train_tens_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    else:
        # Prepcompute dictionary with the unique case ID integers as keys 
        # and a tensor containing the integer indices of the instances 
        # derived from each unique case, within the training dataset, 
        # as values. 
        print("Precomputing dictionary mapping each unique ID to its corresponding indices in training dataset.")
        id_to_indices = precompute_indices(og_caseint_train)

    # Tracking running averages over last ``batch_interval`` batches of each epoch
    # & tracking average validation losses
    train_losses_global = []
    train_losses_act = []
    train_losses_ttne = []

    # Track evolution of validation metrics over the epoch loop by initializing empty lists. 
    avg_MAE_ttne_stand_glob, avg_MAE_ttne_minutes_glob, avg_dam_lev_glob = ([] for _ in range(3))

    # avg_dam_lev_glob, perc_too_early_glob, perc_too_late_glob, perc_correct_glob = ([] for _ in range(4))
    # mean_absolute_length_diff_glob, mean_too_early_glob, mean_too_late_glob = ([] for _ in range(3))

    if remaining_runtime_head:
        avg_MAE_stand_RRT_glob, avg_MAE_minutes_RRT_glob = [], []
    if outcome_bool: 
        avg_BCE_out_glob, avg_auc_roc_glob, avg_auc_pr_glob = [], [], []

    # Creating auxiliary bools 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head

    # Initialize lists for keeping track of training losses of optional 
    # prediction heads if included. 
    if remaining_runtime_head:
        train_losses_rrt = []
    if outcome_bool:
        train_losses_out = []

    # Specifing composite loss function 
    loss_fn = MultiOutputLoss(num_classes, 
                              remaining_runtime_head, 
                              outcome_bool, 
                              clen_dis_ref)

    num_epochs_not_improved = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):

        print(" ")
        print("------------------------------------")
        print('EPOCH {}:'.format(epoch))
        print("____________________________________")

        if clen_dis_ref:
            # CaLenDiR training - UCB Sampling
            print("UCB Sampling...")
            train_sample = sample_train_instances(train_dataset, 
                                                  median_caselen, 
                                                  epoch, 
                                                  id_to_indices)
            # Creating TensorDataset for the training set 
            train_tens_dataset = TensorDataset(*train_sample)

            # Setting seed for reproducable shuffling each epoch
            torch.manual_seed(epoch) 
            train_dataloader = DataLoader(train_tens_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        else: 
            # Setting seed for reproducable shuffling each epoch
            torch.manual_seed(epoch) 

        # Activate gradient tracking
        model.train(True)
        model, optimizer, last_running_avgs = train_epoch(model, 
                                                          train_dataloader, 
                                                          remaining_runtime_head,
                                                          outcome_bool,  
                                                          optimizer, 
                                                          loss_fn, 
                                                          batch_interval, 
                                                          epoch, 
                                                          max_norm)
        train_losses_global.append(last_running_avgs[0])
        train_losses_act.append(last_running_avgs[1])
        train_losses_ttne.append(last_running_avgs[2])
        last_loss = last_running_avgs[-1]
        if only_rrt:
            train_losses_rrt.append(last_running_avgs[3])
        elif only_out:
            train_losses_out.append(last_running_avgs[3])
        elif both:
            train_losses_rrt.append(last_running_avgs[3])
            train_losses_out.append(last_running_avgs[4])

        # Set the model to evaluation mode and disabling dropout
        model.eval()

        if clen_dis_ref:
            # Second tuple element contains list of CB metrics
            _, inf_results, _ = inference_loop(model=model, 
                                               inference_dataset=val_dataset,
                                               remaining_runtime_head=remaining_runtime_head, 
                                               outcome_bool=outcome_bool, 
                                               num_categoricals_pref=num_categoricals_pref, 
                                               mean_std_ttne=mean_std_ttne, 
                                               mean_std_tsp=mean_std_tsp, 
                                               mean_std_tss=mean_std_tss, 
                                               mean_std_rrt=mean_std_rrt, 
                                               og_caseint=og_caseint_val,
                                               results_path=None, 
                                               val_batch_size=4096)
        else: 
            # First tuple element contains list of IB (default) metrics
            inf_results, _, _ = inference_loop(model=model, 
                                               inference_dataset=val_dataset,
                                               remaining_runtime_head=remaining_runtime_head, 
                                               outcome_bool=outcome_bool, 
                                               num_categoricals_pref=num_categoricals_pref, 
                                               mean_std_ttne=mean_std_ttne, 
                                               mean_std_tsp=mean_std_tsp, 
                                               mean_std_tss=mean_std_tss, 
                                               mean_std_rrt=mean_std_rrt, 
                                               og_caseint=og_caseint_val,
                                               results_path=None, 
                                               val_batch_size=4096)

        # TTNE MAE metrics
        avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
        # Average Normalized Damerau-Levenshtein similarity Activity Suffix 
        # prediction
        avg_dam_lev = inf_results[2]


        # # Percentage of validation instances for which the END token was 
        # # predicted too early. 
        # perc_too_early = inf_results[3]
        # # Percentage of validation instances for which the END token was 
        # # predicted too late. 
        # perc_too_late = inf_results[4]
        # # Percentage of validation instances for which the END token was 
        # # predicted at the right moment. 
        # perc_correct = inf_results[5]
        # # Mean absolute lenght difference between predicted and actual 
        # # suffix. 
        # mean_absolute_length_diff = inf_results[6]
        # # Avg num events that END token was predicted too early, averaged 
        # # over all instances for which END was predicted too early. 
        # mean_too_early = inf_results[7]
        # # Avg num events that END token was predicted too late, averaged 
        # # over all instances for which END was predicted too late. 
        # mean_too_late = inf_results[8]

        
        if only_rrt:
            # MAE standardized RRT predictions 
            avg_MAE_stand_RRT = inf_results[3]
            # MAE RRT converted to minutes
            avg_MAE_minutes_RRT = inf_results[4]
        
        elif only_out:
            # Binary Cross Entropy outcome prediction
            avg_BCE_out = inf_results[3]
            # AUC-ROC outcome prediction
            auc_roc = inf_results[4]
            # AUC-PR outcome prediction
            auc_pr = inf_results[5]
        
        elif both:
            # MAE standardized RRT predictions 
            avg_MAE_stand_RRT = inf_results[3]
            # MAE RRT converted to minutes
            avg_MAE_minutes_RRT = inf_results[4]
            # Binary Cross Entropy outcome prediction
            avg_BCE_out = inf_results[5]
            # AUC-ROC outcome prediction
            auc_roc = inf_results[6]
            # AUC-PR outcome prediction
            auc_pr = inf_results[7]
        

        better = False
        if avg_MAE_ttne_stand < best_MAE_ttne: 
            better = True
            best_MAE_ttne = avg_MAE_ttne_stand
        if avg_dam_lev > best_DL_sim:
            better = True 
            best_DL_sim = avg_dam_lev
        # if avg_inference_CE < best_CE:
        #     better = True 
        #     best_CE = avg_inference_CE
        print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
        # print("Avg MAE Type 2 TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE2_stand, avg_MAE2_minutes))
        # print("Avg Cross Entropy acitivty suffix prediction validation set: {}".format(avg_inference_CE))
        print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
        # print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; too late - {}".format(perc_too_early, perc_correct, perc_too_late))
        # print("Too early instances - avg amount of events too early: {}".format(mean_too_early))
        # print("Too late instances - avg amount of events too late: {}".format(mean_too_late))
        # print("Avg absolute amount of events predicted too early / too late: {}".format(mean_absolute_length_diff))
        if remaining_runtime_head: 
            if avg_MAE_stand_RRT < best_MAE_rrt: 
                better = True
                best_MAE_rrt = avg_MAE_stand_RRT
            print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))
            # Store evolution validation measures RRT: 
            avg_MAE_stand_RRT_glob.append(avg_MAE_stand_RRT)
            avg_MAE_minutes_RRT_glob.append(avg_MAE_minutes_RRT)
        if outcome_bool:
            if avg_BCE_out < best_BCE:
                better = True
                best_BCE = avg_BCE_out
            if auc_pr > best_auc_pr:
                better = True
                best_auc_pr = auc_pr
            print("Avg BCE outcome prediction validation set: {}".format(avg_BCE_out))
            print("AUC-ROC outcome prediction validation set: {}".format(auc_roc))
            print("AUC-PR outcome prediction validation set: {}".format(auc_pr))
            avg_BCE_out_glob.append(avg_BCE_out)
            avg_auc_roc_glob.append(auc_roc)
            avg_auc_pr_glob.append(auc_pr)
        if better == False: 
            num_epochs_not_improved += 1
        else:
            num_epochs_not_improved = 0
        # Store other validation measures
        #   TTNE measures
        avg_MAE_ttne_stand_glob.append(avg_MAE_ttne_stand)
        avg_MAE_ttne_minutes_glob.append(avg_MAE_ttne_minutes)
        avg_dam_lev_glob.append(avg_dam_lev)
        # perc_too_early_glob.append(perc_too_early)
        # perc_too_late_glob.append(perc_too_late)
        # perc_correct_glob.append(perc_correct)
        # mean_absolute_length_diff_glob.append(mean_absolute_length_diff)
        # mean_too_early_glob.append(mean_too_early)
        # mean_too_late_glob.append(mean_too_late)

        # Saving checkpoint every epoch

        model_path = os.path.join(path_name, 'model_epoch_{}.pt'.format(epoch))
        checkpoint = {'epoch:' : epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'loss': last_loss}
        torch.save(checkpoint, model_path)
            
        if lr_scheduler_present:
            # Update the learning rate
            lr_scheduler.step()
            
        torch.cuda.empty_cache()


        if num_epochs_not_improved >= patience:
            print("No improvements in validation loss for {} consecutive epochs. Final epoch: {}".format(patience, epoch))
            break
    # Writing training progress to csv at the end of the current training loop
    results_path = os.path.join(path_name, 'backup_results.csv')
    epoch_list = [i for i in range(len(train_losses_global))]
    if both:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                            'composite training loss' : train_losses_global, 
                            'activity training loss (cross entropy)': train_losses_act, 
                            'time till next event training loss (MAE)': train_losses_ttne, 
                            '(complete) remaining runtime training loss (MAE)': train_losses_rrt, 
                            'outcome prediction training loss (BCE)': train_losses_out,
                            'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                            'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                            'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                            # 'Percentage too early (validation)': perc_too_early_glob,    
                            # 'Percentage correct END prediction (validation)': perc_correct_glob,   
                            # 'Percentage too late (validation)': perc_too_late_glob,   
                            # 'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                            # 'Avg too early (validation)': mean_too_early_glob, 
                            # 'Avg too late (validation)': mean_too_late_glob, 
                            'RRT - standardized MAE validation': avg_MAE_stand_RRT_glob, 
                            'RRT - mintues MAE validation': avg_MAE_minutes_RRT_glob, 
                            'Binary Outcome - BCE validation': avg_BCE_out_glob, 
                            'Binary Outcome - AUC-ROC validation': avg_auc_roc_glob, 
                            'Binary Outcome - AUC-PR validation': avg_auc_pr_glob})
        results.to_csv(results_path, index=False)
    elif only_rrt:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                            'composite training loss' : train_losses_global, 
                            'activity training loss (cross entropy)': train_losses_act, 
                            'time till next event training loss (MAE)': train_losses_ttne, 
                            '(complete) remaining runtime training loss (MAE)': train_losses_rrt, 
                            'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                            'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                            'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                            # 'Percentage too early (validation)': perc_too_early_glob,    
                            # 'Percentage correct END prediction (validation)': perc_correct_glob,   
                            # 'Percentage too late (validation)': perc_too_late_glob,   
                            # 'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                            # 'Avg too early (validation)': mean_too_early_glob, 
                            # 'Avg too late (validation)': mean_too_late_glob, 
                            'RRT - standardized MAE validation': avg_MAE_stand_RRT_glob, 
                            'RRT - mintues MAE validation': avg_MAE_minutes_RRT_glob})
        results.to_csv(results_path, index=False)
    
    elif only_out:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                            'composite training loss' : train_losses_global, 
                            'activity training loss (cross entropy)': train_losses_act, 
                            'time till next event training loss (MAE)': train_losses_ttne,
                            'outcome prediction training loss (BCE)': train_losses_out,
                            'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                            'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                            'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                            # 'Percentage too early (validation)': perc_too_early_glob,    
                            # 'Percentage correct END prediction (validation)': perc_correct_glob,   
                            # 'Percentage too late (validation)': perc_too_late_glob,   
                            # 'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                            # 'Avg too early (validation)': mean_too_early_glob, 
                            # 'Avg too late (validation)': mean_too_late_glob,
                            'Binary Outcome - BCE validation': avg_BCE_out_glob, 
                            'Binary Outcome - AUC-ROC validation': avg_auc_roc_glob, 
                            'Binary Outcome - AUC-PR validation': avg_auc_pr_glob})
        results.to_csv(results_path, index=False)
    else:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                            'composite training loss' : train_losses_global, 
                            'activity training loss (cross entropy)': train_losses_act, 
                            'time till next event training loss (MAE)': train_losses_ttne,
                            'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                            'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                            'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                            # 'Percentage too early (validation)': perc_too_early_glob,    
                            # 'Percentage correct END prediction (validation)': perc_correct_glob,   
                            # 'Percentage too late (validation)': perc_too_late_glob,   
                            # 'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                            # 'Avg too early (validation)': mean_too_early_glob, 
                            # 'Avg too late (validation)': mean_too_late_glob
                            })
        results.to_csv(results_path, index=False)