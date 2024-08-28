# *CaLenDiR*-PPM

Welcome to the official repository containing the entire source code for the ***CaLenDiR PPM framework***, proposed in the paper **CaLenDiR: Mitigating Case-Length Distortion in Deep-Learning-Based Predictive Process Monitoring**, submitted to the [ML4PM](https://ml4pm.di.unimi.it/) workshop, part of the [ICPM 2024](https://icpmconference.org/2024/) conference. This framework is specifically designed to mitigate case-length distortion in Deep Learning-based Predictive Process Monitoring (PPM), both during training and evaluation. The **CaLenDiR training** procedure consists of **Uniform Case-Based Sampling (UCBS)**, and in the case of suffix prediction, it additionally incorporates **Suffix-Length-Normalized Loss Functions**. For evaluation, the framework employs **Case-Based Metrics** to accurately reflect the model's performance across the true distribution of case lengths.

As mentioned in the paper, the experimental setup extends on the setup from our previous paper, ***SuTraN: an Encoder-Decoder Transformer for FullContext-Aware Suffix Prediction of Business Processes***, accepted at the **ICPM 2024 main track**. The corresponding meticulously documented repository, [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main) , including all reimplementations, is made available [here](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main). 

___
> __Awaiting publication, a preprint of that paper is provided at the root of this repository (`SuTraN_preprint.pdf`).__

_**© 2024 IEEE**. Personal use of this material is permitted. Permission
from IEEE must be obtained for all other uses, in any current or future
media, including reprinting/republishing this material for advertising or
promotional purposes, creating new collective works, for resale or
redistribution to servers or lists, or reuse of any copyrighted
component of this work in other works._
___

This repository is built on top of that foundation, adding and extending functionality to support the **CaLenDiR** framework. Accordingly, given the high similarity with- and the extensive project description of- the [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main) repository, this README focuses primarily on the new functionality introduced for **CaLenDiR** training, including additional modules and enhancements made to existing code to incorporate and enable **CaLenDiR training and evaluation** across all implementations used in the experimental setup. For detailed explanations on components that remain (largely) unchanged, users are referred to the [SuTraN](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main) repository.


<!-- This repository further extends the Accordingly, the code base contained within this repository is highly similar to the [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main) repository,  -->
<!-- This repository also comprises an extensive project description, further detailling all implementations, as well as providing clear instructions on how to get started with the code.  -->

<!-- Accordingly, 


The **CaLenDiR-PPM** repository extends the experimental setup presented in our previous work on **SuTraN** by introducing the **CaLenDiR PPM framework**. This framework is specifically designed to mitigate case-length distortion in Deep Learning-based Predictive Process Monitoring (PPM). 
As described in the paper, **CaLenDiR training** introduces a comprehensive framework to address case-length distortion. This framework primarily consists of **Uniform Case-Based Sampling (UCBS)** to prevent distortion during training by ensuring balanced contributions from all cases. Specifically for the task of suffix prediction, it further incorporates **Suffix-Length-Normalized Loss Functions** to mitigate additional distortions caused by varying case lengths. For evaluation, the framework employs **Case-Based Metrics** to accurately reflect the model's performance across the true distribution of case lengths.

Given the extensive documentation and the comprehensive codebase already available in the **SuTraN** repository ([SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main)), this README focuses primarily on the new functionality introduced for CaLenDiR training. For detailed explanations on components shared between CaLenDiR and SuTraN, such as the general architecture, training, and evaluation procedures, we refer users to the SuTraN repository.

Given the extensive documentation and comprehensive codebase already available in the **SuTraN** repository ([SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main)), this repository is built on top of that foundation, adding and extending functionality to support the **CaLenDiR** framework. This README focuses primarily on the new functionality introduced for CaLenDiR training, including additional modules and enhancements made to existing code to incorporate and enable CaLenDiR training and evaluation across all implementations used in the experimental setup. For detailed explanations on components that remain unchanged, such as the general architecture of the SuTraN network, training procedures, and standard evaluation methods, users are referred to the SuTraN repository. -->


### Key Extensions in CaLenDiR

1. **CaLenDiR_Utils Subpackage**:
   - This newly introduced subpackage contains essential modules that implement the core functionalities of the CaLenDiR framework:
     - `case_based_sampling.py`: Implements the UCBS algorithm to ensure balanced training across varying case lengths.
     - `seq2seq_norm_lossfunctions.py`: Contains the Suffix-Length-Normalized Loss Functions tailored for sequence-to-sequence models.
     - `weighted_metrics_utils.py`: Provides utilities for calculating case-based evaluation metrics, ensuring accurate performance assessments.

2. **Utils Subpackage**:
   - To maintain consistency and minimize code duplication, all non-normalized sequential loss functions previously scattered across the various subpackages pertaining to the different implementations, have been centralized into this subpackage. 
   <!-- This structure mirrors the organization used for CaLenDiR's normalized loss functions. -->

3. **Extended Training and Inference Modules**:
   - The training and inference procedures for various models (e.g., CRTP_LSTM, SuTraN, OneStepAheadBenchmarks, and LSTM_seq2seq) have been extended to support both standard and CaLenDiR training. These enhancements ensure seamless integration of the new techniques without disrupting the existing functionalities. The necessary adjustments are clearly documented within the docstrings of the training and inference functions, as well as in comment lines within the code base itself. 

4. **Data Preprocessing Extensions**:
   - To support UCBS during training, the data preprocessing pipeline has been extended to include the creation of three additional tensors. These tensors store the original case IDs, which are crucial for accurate case-based sampling and evaluation. Please refer to the comprehensive docstrings of the data generating functions contained within the `Preprocessing` package. 

<!-- ___
Adjust this, mention the preprint here. Arvix. 
A preprint of the accepted ***SuTraN*** paper, will also be communicated at the same day of the release of the ***CaLenDiR-PPM*** repository (*Wednesday 28/08/2024*). 
___ -->


<!-- ___
### Under construction...
___
This repository is set to contain the entire source code for the ***CaLenDiR PPM framework***, proposed in the paper **CaLenDiR: Mitigating Case-Length Distortion in Deep-Learning-Based Predictive Process Monitoring**, submitted to the [ML4PM](https://ml4pm.di.unimi.it/) workshop, part of the [ICPM 2024](https://icpmconference.org/2024/) conference. 

At the moment, to support the integration of the ***CaLenDiR*** framework into other research projects and contribute to the advancement of the PPM field, the authors are finetuning the documentation of the code, as well as the detailled project description (README) containing supplementary materials and detailed instructions on how to leverage the ***CaLenDir*** framework for Deep-Learning-based Predictive Process Monitoring. 

> The complete repository, including extensive documentation and further instructions, is set to be released on **Wednesday 28/08/2024**. 

As mentioned in the paper, the experimental setup extends on the setup from our previous paper, ***SuTraN: an Encoder-Decoder Transformer for FullContext-Aware Suffix Prediction of Business Processes***, accepted at the **ICPM 2024 main track**. The meticulously documented repository, [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main) , including all reimplementations, is already made available [here](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main). This repository also comprises an extensive project description, further detailling all implementations, as well as providing clear instructions on how to get started with the code. A preprint of the accepted ***SuTraN*** paper, will also be communicated at the same day of the release of the ***CaLenDiR-PPM*** repository (*Wednesday 28/08/2024*).  -->


<!-- ### 1. **CaLenDiR_Utils Subpackage**

The training functionality  -->

### Reproducing the Experiments & Getting Started 
The [Reproducing the Experiments section](https://github.com/BrechtWts/SuffixTransformerNetwork#reproducing-the-experiments) of the [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main) repository provided detailled instructions on how to reproduce the experimental setup, for all implementations, of the corresponding paper. In particular, a dedicated end-to-end function called `train_eval()` is created for each model included in the experimental setup. The table underneath lists the different implementations, together with the corresponding modules containing the `train_eval()` for the six different suffix prediction networks. 

|    | Implementation   | Module     | 
| ---| :-----------: |:-------------:| 
|  1 | [***SuTraN*** ](#sutran)     | ``TRAIN_EVAL_SUTRAN_DA.py`` | 
|  2 | [***SuTraN (NDA)***](#sutran)      | ``TRAIN_EVAL_SUTRAN_DA.py`` | 
|  3 | [*CRTP-LSTM*](#crtp-lstm)    | ``TRAIN_EVAL_CRTP_LSTM_DA.py`` | 
|  4 | [*CRTP-LSTM (NDA)*](#crtp-lstm)     | ``CRTP_LSTM`` | 
|  5 | [*ED-LSTM*](#ed-lstm)    | ``TRAIN_EVAL_ED_LSTM.py`` |  
|  6 | [*SEP-LSTM*](#ed-lstm) | ``TRAIN_EVAL_SEP_LSTM.py``     |  

These functions execute the training procedure, and automatically evaluate the models' on the test set after training has finished. 

The only difference with the `train_eval()` functions defined in the [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork#reproducing-the-experiments) repository, is is that they require two additional parameters as input:
1. `clen_dis_ref` - boolean parameter: When set to `True`, the models are trained utilizing the **CaLenDiR**-training procedure. If `False`, they are trained using the standard training procedure. 
1. `median_caselen` - integer parameter: Representing the median case length among the original cases assigned to the training log (post-preprocessing). The values for the three publicly available event logs included in the experiments are:
   1. BPIC17: `34` 
   1. BPIC17-DR: `21` 
   1. BPIC19: `5`
   

As in the [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork#reproducing-the-experiments) repository, predefined data creation functions have to be executed prior to the execution of the `train_eval()` function. These functions are located in the `create_BPIC17_DR_data.py`, `create_BPIC17_OG_data.py` and `create_BPIC19_data.py` modules, pertaining to the BPIC17-DR, BPIC17 and BPIC19 event logs respectively. These data-creating functions write away several data files to disk at predefined locations appended to the root of the repository. The `train_eval()` functions will automatically read the data back into memory, without the need for manual interference. 

Following the detailed step-by-step guide provided in the [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork#reproducing-the-experiments) repository, while additionally specifying the two additional parameters as described here above, will enable you to reproduce the experiments. 

*The final evaluation results are printed written to disk in dedicated subfolders created at the root of the log-specific subfolder (which is created after executing the corresponding data creating functions as discussed above), of which the names are self-explanatory (and can furthermore be derived from the code itself). Regardless of the training procedure used (CaLenDiR training vs standard training), both the instance-based (IB) and case-based (CB) evaluation metrics are printed and written to disk.*

___
> Finally, it is worth noting that the model-specific end-to-end `train_eval()` functions, consolidate all the needed training and evaluation functions, thereby serving as a practical guide to get familiar with the code-base in an end-to-end way, its intended use, and the interplay between the different modules. In combination with the extensive docstrings, detailling the functionality of each function, as well as the input parameters required by each function, efficient re-use of the open-source **CaLenDiR** code base is guaranteed. 

> The exact specification of the `train_eval()` functions and their arguments are provided at the bottom of the six model-specific modules, commented out. 
___


<!-- The experiments pertaining to the **CaLenDiR** paper can be reproduced in an highly similar manner. The sole exception is that the `train_eval()` functions for each model require an additional boolean parameter `clen_dis_ref` as input. When set to `True`, 
Reproducing the experiments for all implementations can be done in a way almost identical  -->

<!-- ### Getting started -->



<!-- ### 3. **Extended Training and Inference Modules**

Compared to the [SuffixTransformerNetwork](https://github.com/BrechtWts/SuffixTransformerNetwork/tree/main) repository,  main training functions pertaining to the different implementations have be -->
