import numpy as np
from edl_pytorch import NormalInvGamma, evidential_regression
import copy
import time
import torch
import csv
from model import *
from BM import *
from datasets import *
from loss import *
from util import *
from config import get_args
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


class ML_NN(nn.Module):
    def __init__(self, in_size = 2,hidden_size = 1024,\
                out_size = 2,n_comp=20,embed = 50,\
                drop_p=0.1,activation = 'softplus',\
                decoder_type = 'deep', cons = False):
        super(ML_NN, self).__init__()
        self.hidden = hidden_size
        hidden =self.hidden
        self.n_comp = n_comp # number of components
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size # number of labels
        self.drop_p = drop_p
        self.activation = activation # use softplus unless specified
        self.decoder_type = decoder_type # test deep and shallow 
        self.cons = cons # whether softplus pi
        if(self.activation =='softplus'):
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden),
                nn.Softplus(),
                nn.Linear(hidden,embed),
            )
            self.decoder1 = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.Softplus(),
                nn.Dropout(p=self.drop_p),
                NormalInvGamma(hidden,n_comp),
            )
            if (decoder_type == 'deep') :
                self.decoder2 = nn.Sequential(
                    nn.Linear(embed, hidden),
                    nn.Softplus(),
                    nn.Linear(hidden, hidden),
                    nn.Softplus(),
                    nn.Linear(hidden, hidden),
                    nn.Softplus(),
                    nn.Linear(hidden, hidden),
                    nn.Softplus(),
                    nn.Dropout(p=self.drop_p),
                    nn.Linear(hidden,out_size),
                    nn.Softplus(),
                )
        self.comp_0 = torch.ones(2*n_comp*out_size).to(device) # components, udpated outside when initialized
        self.comp_1 = torch.zeros(2*n_comp*out_size).to(device) # initilized as zeros for later use

    def forward(self, data):
        # common embeddings
        emb = self.encoder(data)
        # prediction of weight assignments pi from decoder1
        pred_pi = self.decoder1(emb)
        # prediction of instance-wise component adjustment from decoder2
        pred_comp = self.decoder2(emb)

 
        output = pred_comp      
        return output
    def update_comp(self,comp):
        self.comp_0 = self.comp_0+torch.tensor(comp,dtype=torch.float32).to(device)
    def deupdate_comp(self,comp):
        self.comp_0 = self.comp_0-torch.tensor(comp,dtype=torch.float32).to(device)
    def compute_cov(self):
        # components convariance for sampling
        print(self.comp_0.shape)
        # get positive and negative pseudo counts (beta parameters)
        a = self.comp_0[:,:self.n_comp*self.out_size].view(self.n_comp,self.out_size)
        b = self.comp_0[:,self.n_comp*self.out_size:].view(self.n_comp,self.out_size)
        # Bernoulli parameter
        theta = a/(a+b)
        self.lamb_k = []
        self.theta_G = []
        for k in range(self.n_comp):
            self.lamb_k.append(torch.diag(theta[k]*(1-theta[k])))
            # print('lambk',self.lamb_k[-1].shape)
            self.theta_G.append(self.lamb_k[-1]+torch.matmul(theta[k],theta[k].T))
            # print('theta_G',self.theta_G[-1].shape)
        return self.theta_G

def train_sep_bm(    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    uncertainty=False,
    bookkeep = False,
    num_l = 13, fname='1'
):
    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_List = []
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            for phase in ["train", "val"]:
                if phase == "train":
                    print("Training...")
                    model.train()  # Set model to training mode
                else:
                    print("Validating...")
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0
                correct = 0

                # Iterate over data.
                comp_sum = 0

                
                for i, (inputs, labels) in enumerate(dataloaders[phase]):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):

                        y = labels.to(device)
                        # print('inputs',inputs)
                        # print('labels',labels)
                        outputs = model(inputs)
                        # print('outputs',outputs)
                        if bookkeep:
                            theta_s = outputs[1].to(device)
                            print('theta_keep',theta_s.shape,theta_s)
                            theta_s = theta_s.sum(dim=0).view(-1,2*num_classes*num_l)
                            comp_sum+=theta_s.detach().cpu().numpy()

                        loss = criterion(outputs, y.float(), epoch,\
                            num_classes=num_classes, num_l = num_l,\
                       annealing_step=10, device=device,model=model
                        )

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                # preds = predict(model,inputs)
                # match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                # acc = torch.mean(match)
                # print(acc)

                if scheduler is not None:
                    if phase == "train":
                        scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print('epoch loss',epoch_loss)
                loss_List.append(epoch_loss)
                if bookkeep:
                    model.update_comp(comp_sum/ len(dataloaders[phase].dataset))

            print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    comp_save = model.comp_0.cpu().detach().numpy()
    print('comp_save',comp_save)
    file_p = open('./EMLC/EDR/main_res/'+fname+'comp_save_2.npy','wb')
    np.save(file_p,comp_save)
    file_p = open('./EMLC/EDR/main_res/'+fname+'mla_loss_2.npy','wb')
    np.save(file_p,loss_List)
    # metrics = (losses, accuracy)
    return model,loss_List    

def main():
    args,SEED = get_args()

    fname = '0112'+args.ds+args.fname
    fnamesub = fname+'.csv'
    header = ['comps','mse1','mse2','mse3','mse4','train_auc',\
              'train_auc2','train_auc3','test_auc',\
                'test_auc2','test_auc3','micro_auc','micro_auc2',\
                    'micro_auc3','micro_auc4']
    with open('./EMLC/EDR/main_res/'+fnamesub, 'w') as f:
        writer_obj = csv.writer(f)
        writer_obj.writerow(header)

    wnew = args.wnew

    x,y=readDataset(args.ds)
    print(x.shape)
    train = args.train
    pool = args.pool
    test = args.test
    deterministic(SEED)
    train_index,candidate_index,test_index=split(x,label=y,train_rate=train,\
        candidate_rate=pool,test_rate=test,\
            seed=SEED,even=True)
    print('train',len(train_index))
    print('test',len(test_index))
    print('pool',len(candidate_index))
    loss_all = []
    if (args.bookkeep == 'sepopt'):
        bk = True
    for iter_al in range(args.AL_rounds):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        x_can = x[candidate_index]
        y_can = y[candidate_index]


        components_to_test = args.n_components

        model = BernoulliMixture(components_to_test, 200)
        
        drop_p=0.1
        model.fit(y_train)
        score = model.score(y_test)
        
        
        cluster_ids = np.arange(0, components_to_test)
        
        pi = model.pi
        
        mu = model.mu

        np.save('./EMLC/EDR/main_res/'+fname+'mu.npy',mu)
        mu_a = np.concatenate([model.mu,1-model.mu]) 

        comp0 = mu_a.reshape(1,-1)*len(x_train)
        file_p = open('./EMLC/EDR/main_res/'+fname+'comp_0_2.npy','wb')
        np.save(file_p,comp0)
        num_classes = components_to_test
        alpha_t = np.ones((num_classes,len(x)))
        for j in range(len(x)):
            xsol2,lossres=opt_alpha(mu,x[j,:],y[j,:],num_classes)
            alpha_t[:,j]=xsol2
        print('alpha_t',alpha_t)
        print('mu',mu)
        alpha_t = alpha_t.T
        np.save('./EMLC/EDR/main_res/'+fname+'alpha_t.npy',alpha_t)

        alpha_train = alpha_t[train_index]
        alpha_test = alpha_t[test_index]
        np.save('./EMLC/EDR/main_res/'+fname+'alpha_train.npy',alpha_train)

        np.save('./EMLC/EDR/main_res/'+fname+'alpha_test.npy',alpha_test)

        # alpha_train = alpha_t[train_ind]
        # alpha_test = alpha_t[test_ind]
        alphatrain = torch.tensor(alpha_train)
        alphatest = torch.tensor(alpha_test)
        xtrain = torch.tensor(x_train)
        xtest = torch.tensor(x_test)
        ytrain = torch.tensor(y_train)
        ytest = torch.tensor(y_test)
        print('ytest',ytest.shape)
        ycan = torch.tensor(y_can)
        xcan = torch.tensor(x_can)
        from sklearn.kernel_ridge import KernelRidge
        krr = KernelRidge(kernel='rbf',alpha=0.10)
        from sklearn.metrics import mean_squared_error
        krr.fit(x_train,alpha_train)

        krr_pred = krr.predict(x)
        # al_train = krr.predict(x_train)
        # mse1 = mean_squared_error(alpha_train,al_train)
        # print("mse1",mse1)
        # al_test=krr.predict(x_test)
        # mse4 = mean_squared_error(alpha_test,al_test)



        trainData=myDataset(xtrain.to(device),ytrain.to(device))
        testData=myDataset(xtest.to(device),ytest.to(device))
        train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
        test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
        dataloaders = {
        "train": train_dataloader,
        "val": test_dataloader,
    }   
    #     dataloaders = {
    #     "train": trainData,
    #     "val": testData,
    # }
        num_l = y.shape[1]
        out_size = num_l
        num_feature = x.shape[1]
        in_size = num_feature
        bm_model = ML_NN(in_size = in_size,\
            hidden_size = args.m_hidden,out_size = out_size,\
            n_comp=num_classes,embed = args.m_embed,\
            cons=args.m_cons,\
            drop_p = args.m_drop_p, activation=args.m_activation,\
            decoder_type = args.m_decoder_type).to(device)

        # bm_model = BM_NN(in_size = in_size,hidden_size = 64,out_size = out_size,\
        #     n_comp=num_classes,embed = 64).to(device)
        pytorch_total_params = sum(p.numel() for p in bm_model.parameters())

        print(" Number of Parameters: ", pytorch_total_params)

        if (args.optimizer=='adam'):
            optimizer = optim.Adam(bm_model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        # optimizer = optim.SGD(bm_model.parameters(), lr=1e-2, momentum=0.9)    
        # criterion = deterministic_loss
        # criterion = evidential_bm_loss
        # criterion = BM_NON_loss


        #weights pretraining


        criterion = ML_NN_loss

        
        bm_model.update_comp(comp0)
        
        model_opt,loss_opt = train_sep_bm(    bm_model,
        dataloaders,
        num_classes,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=args.pretrain_epochs,
        device=None,
        uncertainty=False,
        bookkeep=False,
        num_l = num_l,
        fname=fname
        )
        model_opt.eval()
        # output_t = model_opt(xtest.float().to(device))
        loss_all+=loss_opt
        output_t = model_opt(xtest.float().to(device))
        y1=y_test
        y1s = np.sum(y1,0)
        col = np.where(y1s!=0)[0]
        y2 = y1[:,col]
        pred12 = output_t[:,col].cpu().detach().numpy()



        test_auc_micro = metrics.roc_auc_score(y2,pred12,average='micro')
        test_auc_macro = metrics.roc_auc_score(y2,pred12,average='macro')
        print('test mi:alpha1',test_auc_micro)
        print('test ma:alpha1',test_auc_macro)

if __name__ == "__main__":
    main() 