"""Functionality training procedure CRTP-LSTM benchmark. 
"""
import torch
import torch.nn as nn
from CRTP_LSTM.train_utils_lstm import MultiOutputLoss
from CRTP_LSTM.train_utils_lstm_masked import MaskedMultiOutputLoss
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from CRTP_LSTM.inference_procedure_lstm import inference_loop

# Importing functionality for Uniform Case-Based Sampling (UCBS) 
# (part of CaLenDiR training)
from CaLenDiR_Utils.case_based_sampling import sample_train_instances, precompute_indices

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_epoch(model, 
                training_loader, 
                optimizer,
                loss_fn, 
                batch_interval,
                epoch_number, 
                max_norm):

    # Tracking global loss over all prediction heads:
    running_loss_glb = []
    # Tracking loss of each prediction head separately: 
    running_loss_act = [] # Cross-Entropy
    running_loss_rrt = [] # MAE

    # Tracking gradient norms
    original_norm_glb = []
    clipped_norm_glb = []


    for batch_num, data in tqdm(enumerate(training_loader), desc="Batch calculation at epoch {}.".format(epoch_number)): 
        inputs = data[:-3]
        labels = data[-3:]
        # Sending inputs and labels to GPU
        inputs = [input_tensor.to(device) for input_tensor in inputs]
        labels = [label_tensor.to(device) for label_tensor in labels]

        # Restoring gradients to 0 for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Computing composite loss and individual losses for track keeping
        loss_results = loss_fn(outputs, labels)
        
        # Compute the loss and its gradients
        loss = loss_results[0]
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
        running_loss_rrt.append(loss_results[-1])
        
        if batch_num % batch_interval == (batch_interval-1):
                print("------------------------------------------------------------")
                print("Epoch {}, batch {}:".format(epoch_number, batch_num))
                print("Average original gradient norm: {} (over last {} batches)".format(sum(original_norm_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Average clipped gradient norm: {} (over last {} batches)".format(sum(clipped_norm_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Running average global loss: {} (over last {} batches)".format(sum(running_loss_glb[-batch_interval:])/batch_interval, batch_interval))
                print("Running average activity prediction loss: {} (Cross Entropy over last {} batches)".format(sum(running_loss_act[-batch_interval:])/batch_interval, batch_interval))
                print("Running average (complete) remaining runtime prediction loss: {} (MAE over last {} batches)".format(sum(running_loss_rrt[-batch_interval:])/batch_interval, batch_interval))
                print("------------------------------------------------------------")

    print("=======================================")
    print("End of epoch {}".format(epoch_number))
    print("=======================================")
    last_running_avg_glob = sum(running_loss_glb[-batch_interval:])/batch_interval
    print("Running average global loss: {} (over last {} batches)".format(last_running_avg_glob, batch_interval))
    last_running_avg_act = sum(running_loss_act[-batch_interval:])/batch_interval
    print("Running average activity prediction loss: {} (Cross Entropy over last {} batches)".format(last_running_avg_act, batch_interval))

    last_running_avg_rrt = sum(running_loss_rrt[-batch_interval:])/batch_interval
    print("Running average (complete) remaining runtime prediction loss: {} (MAE over last {} batches)".format(last_running_avg_rrt, batch_interval))
    last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_rrt, loss
    return model, optimizer, last_running_avgs
            

def train_model(model, 
                optimizer, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
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
                lr_scheduler_present=False, 
                # masking = True,
                lr_scheduler=None, 
                best_DL_sim = -1, 
                best_MAE_rrt = 1e9, 
                max_norm = 2.):
    """Train CRTP-LSTM benchmark. After every epoch, current model is ran 
    on separate validation set. If the validation loss does not improve for 
    59 epochs, early stopping is triggered. A callback of the model is also 
    stored on disk for every epoch. 

    Parameters
    ----------
    model : CRTP_LSTM
        The initialized and current version of a CRTP-LSTM neural 
        network. Should be in evaluation mode already. 
    optimizer : torch optimizer
        NAdam torch optimizer. Already initialized when feeding it to 
        ``train_model()``.
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
    lr_scheduler_present : bool, optional
        Indicates whether we work with a learning rate scheduler wrapped 
        around the optimizer. If True, learning rate scheduler 
        included. If False (default), not. 
    masking : bool, optional 
        Whether or not the (right-padded) padding tokens for the activity 
        suffix and remaining runtime suffix labels should be padded 
        during training, and hence whether the validation metric should 
        account for the paddings too. `True` by default. 
    lr_scheduler : torch lr_scheduler or None
        If ``lr_scheduler_present=True``, a lr_scheduler that is wrapped 
        around the optimizer should be provided as well. The CRTP-LSTM 
        benchmark should be ran with the 
        `torch.optim.lr_scheduler.ReduceLROnPlateau` learning rate 
        scheduler. 
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
        best results of the previous training loop should be given. 
    max_norm : float, optional
        Max gradient norm used for clipping during training. By default 2.
    """
    if lr_scheduler_present:
        if lr_scheduler==None:
            print("No lr_scheduler provided.")
            return -1, -1, -1, -1

    # Checking whether GPU is used
    print("Device: {}".format(device))

    # Assigning CRTP-LSTM to gpu
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
    train_losses_rrt = []

    avg_dam_lev_glob, avg_MAE_stand_RRT_glob, avg_MAE_minutes_RRT_glob, avg_MAE_ttne_minutes_glob = ([] for _ in range(4))

    # avg_dam_lev_glob, perc_too_early_glob, perc_too_late_glob, perc_correct_glob = ([] for _ in range(4))

    # mean_absolute_length_diff_glob, mean_too_early_glob, mean_too_late_glob = ([] for _ in range(3))

    # avg_MAE_stand_RRT_glob, avg_MAE_minutes_RRT_glob, avg_MAE_ttne_minutes_glob= [], [], []

    val_loss_glob = []
    val_loss_best = 1e9
    # if masking:
    # Only with masking from now on - empirically confirmed to deliver 
    # more competitive results. 

    # Specifing composite loss function 
    loss_fn = MaskedMultiOutputLoss(num_classes, clen_dis_ref)
    # else:
    #     loss_fn = MultiOutputLoss(num_classes)

    # Early stopping with patience 24 epochs for validation loss 
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
                                                #   og_caseint_train, 
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
                                                          optimizer, 
                                                          loss_fn, 
                                                          batch_interval, 
                                                          epoch, 
                                                          max_norm)

        train_losses_global.append(last_running_avgs[0])
        train_losses_act.append(last_running_avgs[1])
        train_losses_rrt.append(last_running_avgs[2])
        last_loss = last_running_avgs[-1]


        # Set the model to evaluation mode and disable dropout
        model.eval()

        if clen_dis_ref:
            # Second tuple element contains list of CB metrics 
            _, inf_results, _ = inference_loop(model=model, 
                                               inference_dataset=val_dataset, 
                                               num_categoricals_pref=num_categoricals_pref,
                                               mean_std_ttne=mean_std_ttne, 
                                               mean_std_tsp=mean_std_tsp, 
                                               mean_std_tss=mean_std_tss, 
                                               mean_std_rrt=mean_std_rrt, 
                                               og_caseint=og_caseint_val,
                                               #  masking=masking, 
                                            #    masking=True,
                                               results_path=None, 
                                               val_batch_size=2048)
        else: 
            # First tuple element contains list of IB (default) metrics 
            inf_results, _, _ = inference_loop(model=model, 
                                               inference_dataset=val_dataset, 
                                               num_categoricals_pref=num_categoricals_pref,
                                               mean_std_ttne=mean_std_ttne, 
                                               mean_std_tsp=mean_std_tsp, 
                                               mean_std_tss=mean_std_tss, 
                                               mean_std_rrt=mean_std_rrt, 
                                               og_caseint=og_caseint_val,
                                               #  masking=masking, 
                                            #    masking=True,
                                               results_path=None, 
                                               val_batch_size=2048)

        # Average Normalized Damerau-Levenshtein similarity Activity Suffix 
        # prediction
        avg_dam_lev = inf_results[0]

        # # Percentage of validation instances for which the END token was 
        # # predicted too early. 
        # perc_too_early = inf_results[1]
        # # Percentage of validation instances for which the END token was 
        # # predicted too late. 
        # perc_too_late = inf_results[2]
        # # Percentage of validation instances for which the END token was 
        # # predicted at the right moment. 
        # perc_correct = inf_results[3]
        # # Mean absolute lenght difference between predicted and actual 
        # # suffix. 
        # mean_absolute_length_diff = inf_results[4]
        # # Avg num events that END token was predicted too early, averaged 
        # # over all instances for which END was predicted too early. 
        # mean_too_early = inf_results[5]
        # # Avg num events that END token was predicted too late, averaged 
        # # over all instances for which END was predicted too late. 
        # mean_too_late = inf_results[6]

        # Evaluation RRT metrics: only based on first remaining runtime predictions
        #   - standardized
        avg_MAE_stand_RRT = inf_results[1]
        #   - minutes
        avg_MAE_minutes_RRT = inf_results[2]

        # validation loss 
        val_loss = inf_results[3]

        # Timestamp suffix validation loss 
        avg_MAE_ttne_minutes = inf_results[4]
        
        

        if avg_dam_lev > best_DL_sim:
            best_DL_sim = avg_dam_lev
        print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
        # print("Percentage of suffixes predicted to END: too early - {} ; right moment - {} ; too late - {}".format(perc_too_early, perc_correct, perc_too_late))
        # print("Too early instances - avg amount of events too early: {}".format(mean_too_early))
        # print("Too late instances - avg amount of events too late: {}".format(mean_too_late))
        # print("Avg absolute amount of events predicted too early / too late: {}".format(mean_absolute_length_diff))
        print("Avg MAE TTNE prediction validation set: {} (minutes)'".format(avg_MAE_ttne_minutes))
        if avg_MAE_stand_RRT < best_MAE_rrt: 
            best_MAE_rrt = avg_MAE_stand_RRT
        print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))
        print("Avg validation loss: {}".format(val_loss))
        # Store evolution validation measures RRT: 
        avg_MAE_stand_RRT_glob.append(avg_MAE_stand_RRT)
        avg_MAE_minutes_RRT_glob.append(avg_MAE_minutes_RRT)

        if val_loss < val_loss_best:
            num_epochs_not_improved = 0
            val_loss_best = val_loss
        elif val_loss > val_loss_best:
            num_epochs_not_improved += 1

        # Store other validation measures
        avg_dam_lev_glob.append(avg_dam_lev)
        # perc_too_early_glob.append(perc_too_early)
        # perc_too_late_glob.append(perc_too_late)
        # perc_correct_glob.append(perc_correct)
        # mean_absolute_length_diff_glob.append(mean_absolute_length_diff)
        # mean_too_early_glob.append(mean_too_early)
        # mean_too_late_glob.append(mean_too_late)
        val_loss_glob.append(val_loss)
        avg_MAE_ttne_minutes_glob.append(avg_MAE_ttne_minutes)

        # Saving checkpoint after every epoch
        model_path = os.path.join(path_name, 'model_epoch_{}.pt'.format(epoch))
        checkpoint = {'epoch:' : epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'loss': last_loss}
        torch.save(checkpoint, model_path)
        
        if lr_scheduler_present:
            # Update the learning rate. For the CRTP-LSTM benchmark, 
            # this depends on the validation loss. 
            lr_scheduler.step(val_loss)

        # Empty redundant cache on GPU
        torch.cuda.empty_cache()
        
        # patience of 59 as defined by the authors of the original CRTP-LSTM implementation
        if num_epochs_not_improved == 59:
            print("No improvements in validation loss for 59 consecutive epochs. Final epoch: {}".format(epoch))
            break

    # Writing training progress to csv at the end of the current training loop
    results_path = os.path.join(path_name, 'backup_results.csv')
    epoch_list = [i for i in range(len(train_losses_global))]
    results = pd.DataFrame(data = {'epoch' : epoch_list, 
                        'composite training loss' : train_losses_global, 
                        'activity training loss (cross entropy)': train_losses_act, 
                        '(complete) remaining runtime training loss (suffix) (MAE)': train_losses_rrt, 
                        'composite validation loss' : val_loss_glob,
                        'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                        # 'Percentage too early (validation)': perc_too_early_glob,    
                        # 'Percentage correct END prediction (validation)': perc_correct_glob,   
                        # 'Percentage too late (validation)': perc_too_late_glob,   
                        # 'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                        # 'Avg too early (validation)': mean_too_early_glob, 
                        # 'Avg too late (validation)': mean_too_late_glob, 
                        'TTNE_avg_MAE_minutes' : avg_MAE_ttne_minutes_glob,
                        'RRT - standardized MAE validation': avg_MAE_stand_RRT_glob, 
                        'RRT - mintues MAE validation': avg_MAE_minutes_RRT_glob})
    results.to_csv(results_path, index=False)
    return model, optimizer, epoch, last_loss