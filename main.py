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
                # if bookkeep:
                #     model.update_comp(comp_sum)
                # losses["loss"].append(epoch_loss)
                # losses["phase"].append(phase)
                # losses["epoch"].append(epoch)
                # accuracy["accuracy"].append(epoch_acc.item())
                # accuracy["epoch"].append(epoch)
                # accuracy["phase"].append(phase)

                # print(
                #     "{} loss: {:.4f} acc: {:.4f}".format(
                #         phase.capitalize(), epoch_loss, epoch_acc
                #     )
                # )

                # # deep copy the model
                # if phase == "val" and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(model.state_dict())

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
        al_train = krr.predict(x_train)
        mse1 = mean_squared_error(alpha_train,al_train)
        print("mse1",mse1)
        al_test=krr.predict(x_test)
        mse4 = mean_squared_error(alpha_test,al_test)



        trainData=myDataset(xtrain.to(device),alphatrain.to(device))
        testData=myDataset(xtest.to(device),alphatest.to(device))
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
        bm_model = BM_sep_EDR_NN_custom(in_size = in_size,hidden_size = 1024,out_size = out_size,\
            n_comp=num_classes,embed = 1024,drop_p = drop_p).to(device)

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
        criterion = BM_weiNIG_loss

        # criterion = BM_NIG_loss

        
        bm_model.update_comp(comp0)
        
        model_opt,loss_opt = train_sep_bm(    bm_model,
        dataloaders,
        num_classes,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=5000,
        device=None,
        uncertainty=False,
        bookkeep=False,
        num_l = num_l,
        fname=fname
        )
        model_opt.eval()
        output_t = model_opt(xtest.float().to(device))
        loss_all+=loss_opt
        print('output_t',output_t)
        alpha_t = np.ones((num_classes,len(x_test)))
        for j in range(len(x_test)):
            xsol2,lossres=opt_alpha(mu,x_test[j,:],y_test[j,:],num_classes)
            alpha_t[:,j]=xsol2
        print('alpha_t',torch.tensor(alpha_t))
        print('mu',torch.tensor(mu))
        print('ytest',ytest)

        np.save('./EMLC/EDR/main_res/'+fname+'test_alpha_t.npy',alpha_t)

        mu2 = np.array(mu)
        al1 = np.array(alpha_t)
        al2 = np.array(output_t[0].cpu().detach().numpy())
        np.save('./EMLC/EDR/main_res/'+fname+'test_alpha_1.npy',al1)
        np.save('./EMLC/EDR/main_res/'+fname+'test_alpha_2.npy',al2)

        np.save('./EMLC/EDR/main_res/'+fname+'x_train.npy',x_train)
        np.save('./EMLC/EDR/main_res/'+fname+'x_test.npy',x_test)
        np.save('./EMLC/EDR/main_res/'+fname+'y_train.npy',y_train)
        np.save('./EMLC/EDR/main_res/'+fname+'y_test.npy',y_test)

        #al2 = al2+1
        al1 = al1.T
        pred1 = np.matmul(al1,mu2)
        al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
        pred2 = np.matmul(al2p,mu2)
        # y1 = np.array(ytest)
        y1=y_test
        mse2 = mean_squared_error(al1,al2)
        mse22 = mean_squared_error(alpha_test,al2)

        y1s = np.sum(y1,0)
        col = np.where(y1s!=0)[0]
        print('sum',y1s[col])
        y2 = y1[:,col]
        pred12 = pred1[:,col]
        pred22 = pred2[:,col]
        mse5 = mean_squared_error(al1,alpha_test)

        auc_micro = metrics.roc_auc_score(y2,pred12,average='micro')
        auc_macro = metrics.roc_auc_score(y2,pred12,average='macro')
        print('mi:alpha1',auc_micro)
        print('ma:alpha1',auc_macro)

        auc_micro = metrics.roc_auc_score(y2,pred22,average='micro')
        auc_macro = metrics.roc_auc_score(y2,pred22,average='macro')   
        print('mi:alpha2',auc_micro)
        print('ma:alpha2',auc_macro)


        # al2 = al2+1
        #al1 = al1.T
        # pred1 = np.matmul(al1,mu2)
        pred1 = np.matmul(alpha_test,mu2)

        al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
        pred2 = np.matmul(al2p,mu2)
        # y1 = np.array(ytest)
        y1=y_test
        y1s = np.sum(y1,0)
        col = np.where(y1s!=0)[0]
        y2 = y1[:,col]
        pred12 = pred1[:,col]
        pred22 = pred2[:,col]
        pred3 = np.matmul(al_test,mu2)
        pred32 = pred3[:,col]

        test_auc_micro = metrics.roc_auc_score(y2,pred12,average='micro')
        test_auc_macro = metrics.roc_auc_score(y2,pred12,average='macro')
        print('test mi:alpha1',test_auc_micro)
        print('test ma:alpha1',test_auc_macro)

        test_auc_micro_2 = metrics.roc_auc_score(y2,pred22,average='micro')
        test_auc_macro_2 = metrics.roc_auc_score(y2,pred22,average='macro')   
        print('test mi:alpha2',test_auc_micro_2)
        print('test ma:alpha2',test_auc_macro_2)

        test_auc_micro_3 = metrics.roc_auc_score(y2,pred32,average='micro')
        test_auc_macro_3 = metrics.roc_auc_score(y2,pred32,average='macro')
        print('test mi:alpha3',test_auc_micro_3)
        print('test ma:alpha3',test_auc_macro_3)
        # res_f = quick_test_fixed(model_opt,xtest,ytest,dataloaders,num_classes=num_classes,num_l=num_l,device=device)
        # print(res_f)
        theta_p = output_t[1].detach().cpu().numpy()
        print('theta',theta_p.shape)
        a_p = theta_p[:,:num_classes*num_l]
        b_p = theta_p[:,num_classes*num_l:]
        print('comp_0',model_opt.comp_0, model_opt.comp_0.shape)
        a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
        b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
        a_sl = np.repeat(a_sl,len(a_p),axis=0)
        b_sl = np.repeat(b_sl,len(b_p),axis=0)

        a_new = a_p/wnew+a_sl
        b_new = b_p/wnew+b_sl

        theta_new = a_new/(a_new+b_new)
        print('al2p',al2p.shape)
        pred = np.zeros((len(y_test),y_test.shape[-1]))
        for i in range(len(y_test)):
            pred_0 = np.matmul(al2p[i],theta_new[i].reshape(components_to_test,-1))
            pred[i] = pred_0 
        print('y1',ytest[:10,:])
        print('yt',y_test[:10])
        print('pred',pred[:10,:])
        auc_micro10 = metrics.roc_auc_score(y2,pred[:,col],average='micro')
        np.save('./EMLC/EDR/main_res/'+fname+'y2_sep.npy',y2)
        np.save('./EMLC/EDR/main_res/'+fname+'y_pred_sep.npy',pred[:,col])

        print('mi:alpha10',auc_micro10)
        auc_macro10 = metrics.roc_auc_score(y2,pred[:,col],average='macro')
        print('ma:alpha10',auc_macro10)
        output_t = model_opt(xtrain.float().to(device))

        print('output_t',output_t)
        alpha_t = np.ones((num_classes,len(x_train)))
        for j in range(len(x_train)):
            xsol2,lossres=opt_alpha(mu,x_train[j,:],y_train[j,:],num_classes)
            alpha_t[:,j]=xsol2

        mu2 = np.array(mu)
        al1 = np.array(alpha_t)
        al2 = np.array(output_t[0].cpu().detach().numpy())
        np.save('./EMLC/EDR/main_res/'+fname+'alpha_1.npy',al1)
        np.save('./EMLC/EDR/main_res/'+fname+'alpha_2.npy',al2)

        #al2 = al2+1
        al1 = al1.T
        print('alpha_t',torch.tensor(al1))
        print('mu',torch.tensor(mu))
        print('ytrain',ytrain)
        print("alpha sparse",len(al1),np.sum(al1,axis=0))


        mse3 = mean_squared_error(al1,al2)
        mse33 = mean_squared_error(alpha_train,al2)

        print("mse1",mse1)
        print("mse2",mse2)
        print("mse3",mse3)
        print("mse4",mse4)



        # pred1 = np.matmul(al1,mu2)
        pred1 = np.matmul(alpha_train,mu2)

        al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
        pred2 = np.matmul(al2p,mu2)
        # y1 = np.array(ytrain)
        y1=y_train
        y1s = np.sum(y1,0)
        col = np.where(y1s!=0)[0]
        print('sum',y1s[col])
        y2 = y1[:,col]
        pred12 = pred1[:,col]
        pred22 = pred2[:,col]

        auc_micro = metrics.roc_auc_score(y2,pred12,average='micro')
        auc_macro = metrics.roc_auc_score(y2,pred12,average='macro')
        print('mi:alpha1',auc_micro)
        print('ma:alpha1',auc_macro)

        auc_micro = metrics.roc_auc_score(y2,pred22,average='micro')
        auc_macro = metrics.roc_auc_score(y2,pred22,average='macro')   
        print('mi:alpha2',auc_micro)
        print('ma:alpha2',auc_macro)


        # al2 = al2+1
        #al1 = al1.T
        pred1 = np.matmul(al1,mu2)
        al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
        pred2 = np.matmul(al2p,mu2)
        pred3 = np.matmul(al_train,mu2)

        # y1 = np.array(ytrain)
        y1=y_train
        y1s = np.sum(y1,0)
        col = np.where(y1s!=0)[0]
        y2 = y1[:,col]
        pred12 = pred1[:,col]
        pred22 = pred2[:,col]
        pred32 = pred3[:,col]

        auc_micro = metrics.roc_auc_score(y2,pred12,average='micro')
        auc_macro = metrics.roc_auc_score(y2,pred12,average='macro')
        print('mi:alpha1',auc_micro)
        print('ma:alpha1',auc_macro)

        auc_micro_2 = metrics.roc_auc_score(y2,pred22,average='micro')
        auc_macro_2 = metrics.roc_auc_score(y2,pred22,average='macro')   
        print('mi:alpha2',auc_micro_2)
        print('ma:alpha2',auc_macro_2)
        auc_micro_3 = metrics.roc_auc_score(y2,pred32,average='micro')
        auc_macro_3 = metrics.roc_auc_score(y2,pred32,average='macro')
        print('mi:alpha3',auc_micro_3)
        print('ma:alpha3',auc_macro_3)


        print("mse1",mse1)
        print("mse2",mse2)
        print("mse3",mse3)
        print("mse22",mse22)
        print("mse33",mse33)

        print("mse4",mse4)
        print("mse5",mse5)
        print(fname)
        print('y1',ytest[:10,:])
        print('yt',y_test[:10])


        datapoint = [0, mse1,mse2,mse3,mse4,auc_macro,auc_macro_2,auc_macro_3,test_auc_macro,test_auc_macro_2,test_auc_macro_3,auc_macro10,test_auc_micro,test_auc_micro_2,test_auc_micro_3,auc_micro10]
        with open('./EMLC/EDR/main_res/'+fnamesub,'a') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(datapoint)

if __name__ == "__main__":
    main()    