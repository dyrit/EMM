import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default=2,\
                        help="Seed for the code")
    parser.add_argument('--ds', type= str, default="Delicious", \
                    help="dataset name")

    parser.add_argument('--fname', type= str, default="bs", \
                        help="Identifier For Script")
    parser.add_argument('--lambda_ker', type= float, default=0.1, \
                        help="Kernel Regularization")

    parser.add_argument('--gpu_id', type=int, default=0, \
                        help = "GPU ID")
    parser.add_argument('--debug_id', type=int, default=2, \
                        help = "Debug ID")
    parser.add_argument('--hidden_size', type=int, default=1024, \
                        help = "The hidden layer size")
    parser.add_argument('--seed0', type = int, default=1,\
                        help="Seed for loading")

# AL parameters    
    parser.add_argument('--train', type= float, default=0.01, \
                    help="train size")
    parser.add_argument('--pool', type= float, default=0.6, \
                    help="pool size")
    parser.add_argument('--test', type= float, default=0.2, \
                    help="test size") 

    parser.add_argument('--AL_rounds', type = int, default=10,\
                    help="Number of AL rounds")
    
    parser.add_argument('--n_components', type = int, default=6,\
                    help="Number of mixture components")

    parser.add_argument('--wnew', type= float, default=10.0, \
                    help="Components update weight") 
    
# training parameters
    parser.add_argument('--optimizer', type= str, default="adam", \
                    help="Type of optimizer")
    parser.add_argument('--lr', type= float, default=1e-4, \
                    help="Learning rate")  
    parser.add_argument('--wd', type= float, default=0, \
                    help="Weight decay")  

# experiment settings
    parser.add_argument('--pretrain_loss', type= str, default="NIG", \
                    help="Type of loss used in weights pretraining step")
    parser.add_argument('--pretrain_epochs', type = int, default=5000,\
                    help="Number of weights pretraining epochs")

    args = parser.parse_args()
    return args, args.seed