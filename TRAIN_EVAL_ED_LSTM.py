"""Module containing the entire pipeline to train and evaluate the 
ED-LSTM benchmark. 
"""
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle 
# import sys 




def load_checkpoint(model, path_to_checkpoint, train_or_eval, lr):
    """Loads already trained model into memory with the 
    learned weights, as well as the optimizer in its 
    state when saving the model.

    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 

    Parameters
    ----------
    model : instance of the CRTP Transformer
        Should just be initialized with the correct initialization 
        arguments, like you would have done in the beginning. 
    path_to_checkpoint : string
        Exact path where the checkpoint is stored on disk. 
    train_or_eval : str, {'train', 'eval'}
        Indicating whether you want to resume training ('train') with the 
        loaded model, or you want to evaluate it ('eval'). The layers of 
        the model will be returned in the appropriate mode. 
    lr : float 
        Learning rate of the optimizer last used for training. 
    
    Returns
    -------
    model : ...
        With trained weights loaded. 
    optimizer : ... 
        With correct optimizer state loaded. 
    final_epoch_trained : int 
        Number of final epoch that the model is trained for. 
        If you want to resume training with the loaded model, 
        start from start_epoch = final_epoch_trained + 1. 
    final_loss : ... 
        Last loss of last epoch. Don't think you need it for resuming 
        training. 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)

    if (train_or_eval!= 'train') and (train_or_eval!= 'eval'):
        print("train_or_eval argument should be either 'train' or 'eval'.")
        return -1, -1, -1, -1

    checkpoint = torch.load(path_to_checkpoint)
    # Loading saved weights of the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Loading saved state of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    # optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Loading number of last epoch that the saved model was trained for. 
    final_epoch_trained = checkpoint['epoch:']
    # Last loss of the trained model. 
    final_loss = checkpoint['loss']

    if train_or_eval == 'train':
        model.train()
    else: 
        model.eval()
        
    return model, optimizer, final_epoch_trained, final_loss


def train_eval(log_name, 
               tss_index, 
               median_caselen,
               clen_dis_ref=True):
    """Training and automatically evaluating the ED-LSTM benchmark 
    model with the parameters used in the SuTraN paper. 

    Parameters
    ----------
    log_name : str
        Name of the event log on which the model is trained. Should be 
        the same string as the one specified for the `log_name` parameter 
        of the `log_to_tensors()` function in the 
        `Preprocessing\from_log_to_tensors.py` module. 
    tss_index : int 
        Zero-based index at which the time since start (tss) prefix 
        event feature was stored in the original, fully data-aware 
        datasets. The time since previous event (tsp) prefix event 
        feature is stored at index `tss_index+1`. These two time related 
        features are the only numerical prefix event features retained 
        in the non data-aware benchmark models, and their indices are 
        therefore needed to retrieve these two features. 
    median_caselen : int
        Median case length original cases. 
    clen_dis_ref : bool, optional
        If `True`, Case Length Distribution-Reflective (CaLenDiR) 
        Training is performed. This includes the application of Uniform 
        Case-Based Sampling (UCBS) of instances each epoch, and 
        Suffix-Length-Normalized Loss Functions. If `False`, the default 
        training procedure, in which all instances are used for training 
        each epoch and in which no loss function normalization is 
        performed (and hence in which case-length distortion is not 
        addressed), is performed. `True` by default. 
        Note that the default (non-CaLenDiR) training 
        procedure is performed by choosing the non-default parameter 
        value (`False`) for the boolean `clen_dis_ref` parameter. By 
        retaining the default `True` parameter value for `clen_dis_ref`, 
        the model is trained according to the non-default CaLenDiR 
        training procedure. 
    """

    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            loaded_dict = pickle.load(file)
        
        return loaded_dict


    # -----------------
    temp_string = log_name + '_cardin_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_dict = load_dict(temp_path)
    num_activities = cardinality_dict['concept:name'] + 2
    print(num_activities)

    # cardinality list prefix categoricals 
    temp_string = log_name + '_cardin_list_prefix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_prefix = load_dict(temp_path)

    temp_string = log_name + '_cardin_list_suffix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    # cardinality list suffix categoricals
    cardinality_list_suffix = load_dict(temp_path)

    temp_string = log_name + '_num_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    # To retrieve the number of numerical featrues in the prefix and suffix events respectively 
    num_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_cat_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cat_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_train_means_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_means_dict = load_dict(temp_path)

    temp_string = log_name + '_train_std_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)

    train_std_dict = load_dict(temp_path)

    mean_std_ttne = [train_means_dict['timeLabel_df'][0], train_std_dict['timeLabel_df'][0]]
    mean_std_tsp = [train_means_dict['suffix_df'][1], train_std_dict['suffix_df'][1]]
    mean_std_tss = [train_means_dict['suffix_df'][0], train_std_dict['suffix_df'][0]]
    # mean_std_tss_pref = [train_means_dict['prefix_df'][5], train_std_dict['prefix_df'][5]]
    # mean_std_tsp_pref = [train_means_dict['prefix_df'][6], train_std_dict['prefix_df'][6]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1], train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])
    num_numericals_suf = len(num_cols_dict['suffix_df'])

    num_categoricals_pref, num_categoricals_suf = len(cat_cols_dict['prefix_df']), len(cat_cols_dict['suffix_df'])


    dropout = 0.2
    batch_size = 128


    # # specifying path results and callbacks 
    # backup_path = os.path.join(log_name, "ED_LSTM_results")
    # os.makedirs(backup_path, exist_ok=True)

    # specifying path results and callbacks 
    subfolder_path = os.path.join(log_name, "ED_LSTM_results")
    os.makedirs(subfolder_path, exist_ok=True)

    
    if clen_dis_ref: 
        backup_path = os.path.join(subfolder_path, "CaLenDiR_training")
    else:
        backup_path = os.path.join(subfolder_path, "default_training")
    
    os.makedirs(backup_path, exist_ok=True)

    # Loading train, validation and test sets 
    # Training set 
    temp_path = os.path.join(log_name, 'train_tensordataset.pt')
    train_dataset = torch.load(temp_path)

    # Validation set
    temp_path = os.path.join(log_name, 'val_tensordataset.pt')
    val_dataset = torch.load(temp_path)

    # Test set 
    temp_path = os.path.join(log_name, 'test_tensordataset.pt')
    test_dataset = torch.load(temp_path)

    # For the prefix event token features: slice out only 
    # the activity label, as well as the two time 
    # features (time since start and time since previous)
    # tsp_index = tss_index+1
    # train_dataset = (train_dataset[num_categoricals_pref-1], ) + (train_dataset[num_categoricals_pref][:, :, -2:],) + train_dataset[num_categoricals_pref+1:]

    # test_dataset = (test_dataset[num_categoricals_pref-1], ) + (test_dataset[num_categoricals_pref][:, :, -2:],) + test_dataset[num_categoricals_pref+1:]

    # val_dataset = (val_dataset[num_categoricals_pref-1], ) + (val_dataset[num_categoricals_pref][:, :, -2:],) + val_dataset[num_categoricals_pref+1:]
    exclusive_bound = tss_index+2
    train_dataset = (train_dataset[num_categoricals_pref-1], ) + (train_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + train_dataset[num_categoricals_pref+1:]

    test_dataset = (test_dataset[num_categoricals_pref-1], ) + (test_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + test_dataset[num_categoricals_pref+1:]

    val_dataset = (val_dataset[num_categoricals_pref-1], ) + (val_dataset[num_categoricals_pref][:, :, tss_index:exclusive_bound],) + val_dataset[num_categoricals_pref+1:]

    # Converting the data format needed for the ED-LSTM benchmark 
    from LSTM_seq2seq.convert_tensordata import convert_to_lstm_data
    train_dataset = convert_to_lstm_data(train_dataset, outcome_bool=False,
                                        num_categoricals_pref=1, 
                                        num_numericals_pref=2)

    val_dataset = convert_to_lstm_data(val_dataset, outcome_bool=False,
                                        num_categoricals_pref=1, 
                                        num_numericals_pref=2)

    test_dataset = convert_to_lstm_data(test_dataset, outcome_bool=False,
                                        num_categoricals_pref=1, 
                                        num_numericals_pref=2)

    # Creating TensorDataset for the training set 
    # train_dataset = TensorDataset(*train_dataset)

    temp_path = os.path.join(log_name, 'og_caseint_train.pt')
    og_caseint_train = torch.load(temp_path)

    temp_path = os.path.join(log_name, 'og_caseint_val.pt')
    og_caseint_val = torch.load(temp_path)

    temp_path = os.path.join(log_name, 'og_caseint_test.pt')
    og_caseint_test = torch.load(temp_path)

    # Setting up GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)


    # Initializing model 
    import random
    # Set a seed value
    seed_value = 24

    # Set Python random seed
    random.seed(seed_value)

    # Set NumPy random seed
    np.random.seed(seed_value)

    # Set PyTorch random seed
    torch.manual_seed(seed_value)

    # If you are using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        # Additional settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Import Benchmark model 
    from LSTM_seq2seq.enc_dec_lstm import EncDecLSTM_no_context

    d_model = 64
    model = EncDecLSTM_no_context(num_activities=num_activities, 
                        d_model=d_model, 
                        n_layers=4,
                        dropout=0.2)


    # Assign to GPU 
    model.to(device)

    # Initializing optimizer and learning rate scheduler 
    decay_factor = 0.96
    lr = 0.0002
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_factor)

    
    # Training procedure 
    from LSTM_seq2seq.train_procedure import train_model
    start_epoch = 0
    num_epochs = 200 
    num_classes = num_activities 
    batch_interval = 800

    train_model(model, 
                optimizer, 
                train_dataset, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                num_classes, 
                batch_interval, 
                backup_path, 
                # num_categoricals_pref, 
                1,
                mean_std_ttne, 
                mean_std_tsp, 
                mean_std_tss, 
                mean_std_rrt, 
                batch_size, 
                clen_dis_ref=clen_dis_ref,
                og_caseint_train=og_caseint_train, 
                og_caseint_val=og_caseint_val,
                median_caselen=median_caselen,
                patience=24,
                lr_scheduler_present=True, 
                lr_scheduler=lr_scheduler, 
                best_MAE_ttne = 1e9, 
                best_DL_sim = -1, 
                best_MAE_rrt = 1e9, 
                max_norm = 2.)


    
    # Re-initializing new model after training to load best callback
    model = EncDecLSTM_no_context(num_activities=num_activities, 
                        d_model=d_model, 
                        n_layers=4,
                        dropout=0.2)

    # Assign to GPU 
    model.to(device)

    # Specifying path of csv in which the training and validation results 
    # of every epoch are stored. 
    final_results_path = os.path.join(backup_path, 'backup_results.csv')

    # Determining best epoch based on the validation 
    # scores for RRT and Activity Suffix prediction
    df = pd.read_csv(final_results_path)
    dl_col = 'Activity suffix: 1-DL (validation)'
    rrt_col = 'RRT - mintues MAE validation'
    df['rrt_rank_val'] = df[rrt_col].rank(method='min').astype(int)
    df['dl_rank_val'] = df[dl_col].rank(method='min', ascending=False).astype(int)
    df['summed_rank_val'] = df['rrt_rank_val'] + df['dl_rank_val']

    # Retrieving the row with the best general performance 
    row_with_lowest_loss = df.loc[df['summed_rank_val'].idxmin()]
    # Retrieve the value of the 'epoch' column for that row
    epoch_value = row_with_lowest_loss['epoch']

    # The models are stored with the string underneath
    best_epoch_string = 'model_epoch_{}.pt'.format(int(epoch_value))
    best_epoch_path = os.path.join(backup_path, best_epoch_string)

    # Loadd best model into memory again 
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=best_epoch_path, train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()

    # Running final inference on test set 
    from LSTM_seq2seq.inference_procedure import inference_loop

    # Initializing directory for final test set results 
    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)



    inf_results_IB, inf_results_CB, pref_suf_results = inference_loop(model, 
                                                                      test_dataset, 
                                                                      1, # num_categoricals_pref
                                                                      mean_std_ttne, 
                                                                      mean_std_tsp, 
                                                                      mean_std_tss, 
                                                                      mean_std_rrt, 
                                                                      og_caseint_test, 
                                                                      results_path, 
                                                                      val_batch_size=2048)
    
    #######################################################
    ###########   INSTANCE-BASED (IB) METRICS   ###########
    #######################################################
    avg_MAE_ttne_stand_IB, avg_MAE_ttne_minutes_IB = inf_results_IB[:2]
    avg_dam_lev_IB, avg_MAE_rrt_sum_minutes_IB = inf_results_IB[2:4]

    # Printing averaged results 
    print("INSTANCE-BASED (IB) METRICS:")
    print("IB - Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand_IB, avg_MAE_ttne_minutes_IB))
    print("IB - Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev_IB))
    print("IB - Avg MAE RRT prediction validation set:  {} (minutes)'".format(avg_MAE_rrt_sum_minutes_IB))


    # Retrieving and storing dictionary of the metrics averaged over all 
    # test set instances (prefix-suffix pairs)
    avg_results_dict_IB = {"MAE TTNE minutes" : avg_MAE_ttne_minutes_IB, 
                        "DL sim" : avg_dam_lev_IB, 
                        "MAE RRT minutes" : avg_MAE_rrt_sum_minutes_IB}
    path_name_average_results_IB = os.path.join(results_path, 'averaged_results_IB.pkl')


    #######################################################
    #############   CASE-BASED (CB) METRICS   #############
    #######################################################
    avg_MAE_ttne_stand_CB, avg_MAE_ttne_minutes_CB = inf_results_CB[:2]
    avg_dam_lev_CB, avg_MAE_rrt_sum_minutes_CB = inf_results_CB[2:4]

    # Printing averaged results 
    print("CASE-BASED (CB) METRICS:")
    print("CB - Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand_CB, avg_MAE_ttne_minutes_CB))
    print("CB - Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev_CB))
    print("CB - Avg MAE RRT prediction validation set:  {} (minutes)'".format(avg_MAE_rrt_sum_minutes_CB))


    # Retrieving and storing dictionary of the metrics averaged over all 
    # test set instances (prefix-suffix pairs)
    avg_results_dict_CB = {"MAE TTNE minutes" : avg_MAE_ttne_minutes_CB, 
                        "DL sim" : avg_dam_lev_CB, 
                        "MAE RRT minutes" : avg_MAE_rrt_sum_minutes_CB}
    path_name_average_results_CB = os.path.join(results_path, 'averaged_results_CB.pkl')

    
    # Retrieving and storing the dictionaries with the 
    # averaged results per prefix and suffix length
    results_dict_pref = pref_suf_results[0]
    results_dict_suf = pref_suf_results[1]

    path_name_prefix = os.path.join(results_path, 'prefix_length_results_dict.pkl')
    path_name_suffix = os.path.join(results_path, 'suffix_length_results_dict.pkl')
    with open(path_name_prefix, 'wb') as file:
        pickle.dump(results_dict_pref, file)
    with open(path_name_suffix, 'wb') as file:
        pickle.dump(results_dict_suf, file)
    with open(path_name_average_results_IB, 'wb') as file:
        pickle.dump(avg_results_dict_IB, file)
    with open(path_name_average_results_CB, 'wb') as file:
        pickle.dump(avg_results_dict_CB, file)


#########################
# EXECUTING ALL EXPERIMENTS
#########################
# # BPIC19
# train_eval(log_name='BPIC_19', 
#            tss_index=1, 
#            median_caselen=5, 
#            clen_dis_ref=True)

# # BPIC17 NOLOOPS 
# train_eval(log_name='BPIC_17_DR', 
#            tss_index=5, 
#            median_caselen=21, 
#            clen_dis_ref=True)


# # BPIC17 ORIGINAL 
# train_eval(log_name='BPIC_17', 
#            tss_index=5, 
#            median_caselen=34, 
#            clen_dis_ref=True)

# # BAC (Private)
# train_eval(log_name='BAC', 
#            tss_index=0, 
#            median_caselen=4, 
#            clen_dis_ref=True)