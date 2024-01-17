# For UNT folks
# https://github.com/dyrit/EMLC_main.git
# This repository maintains a currently worked-on code base for the evidential multi-label worke
# To run the code, do 'python main.py' with desired options (some paths may need to be changed)
# Example: weights training only: python main.py --AL_rounds 1 --tr_rounds 0 --pretrain_epochs 500
# Current evidential loss: --pretrain_loss 'NIG'
# To modify and add loss functions, use 'loss.py' (be mindful of the outputs and targets for the loss function)
# To modify the model, use 'model.py'
# To modify the test step, use 'util.py' (current one is messy, maybe only mse(pi) and AUC should be enough)
# Data files should be stored in a dataMats folder
# The .yml file might not work (ran into an error). Packages other than standard torch, numpy, etc.:\
# pip install edl_pytorch
# https://github.com/dyrit/evidential-learning-pytorch.git or teddykoker/evidential-learning-pytorch
# The NIG layer and evidential loss are also from the repo, but they are not complicated if want to modify
# in the train_sep_bm() function, be mindful of the dataloaders and criterion
# components are from BM, callded 'mu' in main.py (should be theta), 
# weight coefficients are from opt_alpha(), called 'alpha' in main.py 
# could use krr to get an idea of fitting