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

        krr_pred = krr.predict(x)
        # al_train = krr.predict(x_train)
        # mse1 = mean_squared_error(alpha_train,al_train)
        # print("mse1",mse1)
        # al_test=krr.predict(x_test)
        # mse4 = mean_squared_error(alpha_test,al_test)



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
        bm_model = BM_sep_EDR_NN_custom(in_size = in_size,\
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
        if (args.pretrain_loss=='NIG'):
            criterion = BM_weiNIG_loss

        # criterion = BM_NIG_loss

        
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
        # print('output_t',output_t)
        # alpha_t = np.ones((num_classes,len(x_test)))
        # for j in range(len(x_test)):
        #     xsol2,lossres=opt_alpha(mu,x_test[j,:],y_test[j,:],num_classes)
        #     alpha_t[:,j]=xsol2
        # print('alpha_t',torch.tensor(alpha_t))
        # print('mu',torch.tensor(mu))
        # print('ytest',ytest)

        # np.save('./EMLC/EDR/main_res/'+fname+'test_alpha_t.npy',alpha_t)
        datapoint = print_res(model_opt, x_test, y_test, mu, fname,num_classes,
              alpha_t, num_l,wnew,n=0,\
                alpha_test=alpha_test, alpha_train= alpha_train,\
                train_index=train_index,test_index=test_index,\
                handle='train_test',pi=None,bs_pred = krr_pred,\
                    ysum=False,\
                xtest=xtest,ytest=ytest,\
                    x_train=x_train,y_train=y_train)
        with open('./EMLC/EDR/main_res/'+fnamesub,'a') as f:
            writer_obj = csv.writer(f)
            writer_obj.writerow(datapoint)

        for iter in range(args.tr_rounds):
            trainData=myDataset(xtrain.to(device),ytrain.to(device))
            testData=myDataset(xtest.to(device),ytest.to(device))
            train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
            test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
            dataloaders = {
            "train": train_dataloader,
            "val": test_dataloader,
        }   

            bm_model = model_opt.to(device)
            # optimizer = optim.Adam(bm_model.parameters(), \
            #                        lr=1e-3, weight_decay=0)
            optimizer = optim.Adam(bm_model.decoder2.parameters(), \
                                   lr=1e-3, weight_decay=0)

            if(args.l_loss=='NON'):
                criterion = BM_sepNON_loss
            model_opt,loss_opt = train_sep_bm(    bm_model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler=scheduler,
            num_epochs=args.l_epochs,
            device=None,
            uncertainty=False,
            bookkeep=True,
            num_l = num_l,
            fname=fname
            )
            loss_all+=loss_opt
            model_opt.eval()
            # output_t = model_opt(xtest.float().to(device))
            datapoint = print_res(model_opt, x_test, y_test, mu, fname,num_classes,
                alpha_t, num_l,wnew,n=iter,\
                    alpha_test=alpha_test, alpha_train= alpha_train,\
                    train_index=train_index,test_index=test_index,\
                    handle='train_test',pi=None,bs_pred = krr_pred,\
                        ysum=False,\
                    xtest=xtest,ytest=ytest,\
                        x_train=x_train,y_train=y_train)
            with open('./EMLC/EDR/main_res/'+fnamesub,'a') as f:
                writer_obj = csv.writer(f)
                writer_obj.writerow(datapoint)

            # weights-only step
                #change data loader (maybe modify later)
            trainData=myDataset(xtrain.to(device),alphatrain.to(device))
            testData=myDataset(xtest.to(device),alphatest.to(device))
            train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
            test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
            dataloaders = {
            "train": train_dataloader,
            "val": test_dataloader,
        }   

            bm_model = model_opt.to(device)

            # bm_model = BM_NN(in_size = in_size,hidden_size = 64,out_size = out_size,\
            #     n_comp=num_classes,embed = 64).to(device)
            pytorch_total_params = sum(p.numel() for p in bm_model.parameters())

            print(" Number of Parameters: ", pytorch_total_params)

            optimizer = optim.Adam(bm_model.parameters(), lr=1e-3, weight_decay=0)

            if(args.pi_loss=='NIG'):
                criterion = BM_weiNIG_loss
            model_opt,loss_opt = train_sep_bm(    bm_model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler=scheduler,
            num_epochs=args.pi_epochs,
            device=None,
            uncertainty=False,
            bookkeep=False,
            num_l = num_l,
            fname=fname
            )
            loss_all+=loss_opt
            model_opt.eval()

            datapoint = print_res(model_opt, x_test, y_test, mu, fname,num_classes,
                alpha_t, num_l,wnew,n=iter,\
                    alpha_test=alpha_test, alpha_train= alpha_train,\
                    train_index=train_index,test_index=test_index,\
                    handle='train_test',pi=None,bs_pred = krr_pred,\
                        ysum=False,\
                    xtest=xtest,ytest=ytest,\
                        x_train=x_train,y_train=y_train)
            with open('./EMLC/EDR/main_res/'+fnamesub,'a') as f:
                writer_obj = csv.writer(f)
                writer_obj.writerow(datapoint)


# update all data and train baseline
        a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
        b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
        # a_sl = np.repeat(a_sl,len(a_p),axis=0)
        # b_sl = np.repeat(b_sl,len(b_p),axis=0)

        comp0 = a_sl/(a_sl+b_sl)
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

        krr.fit(x_train,alpha_train)
        krr_pred = krr.predict(x)

        trainData=myDataset(xtrain.to(device),alphatrain.to(device))
        testData=myDataset(xtest.to(device),alphatest.to(device))
        train_dataloader=DataLoader(trainData, batch_size=500, shuffle=True)
        test_dataloader=DataLoader(testData, batch_size=len(ytest), shuffle=False)
        dataloaders = {
        "train": train_dataloader,
        "val": test_dataloader,
    } 
        output_t = model_opt(xcan.float().to(device))

        mu_can = np.array(output_t[0].cpu().detach().numpy())
        v2_can = np.array(output_t[2].cpu().detach().numpy())
        a2_can = np.array(output_t[3].cpu().detach().numpy())
        b2_can = np.array(output_t[4].cpu().detach().numpy())
        np.save('./EMLC/EDR/main_res/'+fname+'mu_can.npy',mu_can)
        np.save('./EMLC/EDR/main_res/'+fname+'v2_can.npy',v2_can)
        np.save('./EMLC/EDR/main_res/'+fname+'a2_can.npy',a2_can)
        np.save('./EMLC/EDR/main_res/'+fname+'b2_can.npy',b2_can)
        met = args.al_mtd
        # met = 'evi'
        # met = 'evicov'
        # met = 'evicov1'
        # met = 'evicov2'
        if met =='bvs':
            probs_sorted,idxs = output_t[0].sort(descending=True)
            U = probs_sorted[:,0]-probs_sorted[:,1]
            ind_add = list(np.array(candidate_index)[list(U.sort()[1][:100].cpu().numpy())])
            train_index = train_index+ind_add
            for ind_a in ind_add:
                candidate_index.remove(ind_a)
        elif met =='evi':
            np.save('./EMLC/EDR/main_res/'+fname+'probs.npy',output_t[0].detach().cpu().numpy())

            probs = np.mean(b2_can/(v2_can*(a2_can-1)),axis=1)
            np.save('./EMLC/EDR/main_res/'+fname+'var.npy',probs.detach().cpu().numpy())

            ind_add = list(np.array(candidate_index)[list(np.argsort(probs)[::-1][:100])])
            train_index = train_index+ind_add
            print(ind_add)
            for ind_a in ind_add:
                candidate_index.remove(ind_a)
        elif met =='evicov':
            np.save('./EMLC/EDR/main_res/'+fname+'probs.npy',output_t[0].detach().cpu().numpy())
            theta_p = output_t[1].detach().cpu().numpy()
            a_p = theta_p[:,:num_classes*num_l]
            b_p = theta_p[:,num_classes*num_l:]
            a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
            b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
            a_sl = np.repeat(a_sl,len(a_p),axis=0)
            b_sl = np.repeat(b_sl,len(b_p),axis=0)

            a_new = a_p/wnew+a_sl
            b_new = b_p/wnew+b_sl


            theta_new = a_new/(a_new+b_new)
            print('thetashape',theta_new.shape)

            pred = np.zeros((len(y_can),y_can.shape[-1]))
            for i in range(len(y_can)):
                pred_0 = np.matmul(mu_can[i],theta_new[i].reshape(components_to_test,-1))
                pred[i] = pred_0 
            print('predshape',pred.shape)
            cov = np.matmul(pred,pred.T)
            np.save('./EMLC/EDR/main_res/'+fname+'cov_can.npy',cov)

            print('covshape0,cov.shape')
            probs = np.mean(b2_can/(v2_can*(a2_can-1)),axis=1)+np.mean(cov,axis=1)
            np.save('./EMLC/EDR/main_res/'+fname+'var.npy',probs)

            ind_add = list(np.array(candidate_index)[list(np.argsort(probs)[::-1][:100])])
            train_index = train_index+ind_add
            print(ind_add)
            for ind_a in ind_add:
                candidate_index.remove(ind_a)

        elif met =='evicov1':
            np.save('./EMLC/EDR/main_res/'+fname+'probs.npy',output_t[0].detach().cpu().numpy())
            theta_p = output_t[1].detach().cpu().numpy()
            a_p = theta_p[:,:num_classes*num_l]
            b_p = theta_p[:,num_classes*num_l:]
            a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
            b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
            a_sl = np.repeat(a_sl,len(a_p),axis=0)
            b_sl = np.repeat(b_sl,len(b_p),axis=0)

            a_new = a_p/wnew+a_sl
            b_new = b_p/wnew+b_sl


            theta_new = a_new/(a_new+b_new)

            pred = np.zeros((len(y_can),y_can.shape[-1]))
            cov1 = np.zeros((len(y_can),y_can.shape[-1]))
            for i in range(len(y_can)):
                pred_0 = np.matmul(mu_can[i],theta_new[i].reshape(components_to_test,-1))
                pred[i] = pred_0 
            print('predshape',pred.shape)
            cov = np.matmul(pred,pred.T)
            np.save('./EMLC/EDR/main_res/'+fname+'cov_can.npy',cov)

            print('covshape0,cov.shape')
            probs = np.mean(b2_can/(v2_can*(a2_can-1)),axis=1)+np.mean(cov,axis=1)
            np.save('./EMLC/EDR/main_res/'+fname+'var.npy',probs)

            ind_add = list(np.array(candidate_index)[list(np.argsort(probs)[::-1][:100])])
            train_index = train_index+ind_add
            print(ind_add)
            for ind_a in ind_add:
                candidate_index.remove(ind_a)
        elif met =='evicov2':
            np.save('./EMLC/EDR/main_res/'+fname+'probs.npy',output_t[0].detach().cpu().numpy())
            theta_p = output_t[1].detach().cpu().numpy()
            a_p = theta_p[:,:num_classes*num_l]
            b_p = theta_p[:,num_classes*num_l:]
            a_sl = model_opt.comp_0[:,:num_classes*num_l].detach().cpu().numpy()
            b_sl = model_opt.comp_0[:,num_classes*num_l:].detach().cpu().numpy()
            a_sl = np.repeat(a_sl,len(a_p),axis=0)
            b_sl = np.repeat(b_sl,len(b_p),axis=0)

            a_new = a_p/wnew+a_sl
            b_new = b_p/wnew+b_sl


            theta_new = a_new/(a_new+b_new)

            pred = np.zeros((len(y_can),y_can.shape[-1]))
            cov1 = np.zeros((len(y_can)))
            cov2 = np.zeros((len(y_can),y_can.shape[-1],y_can.shape[-1]))
            for i in range(len(y_can)):
                pred_0 = np.matmul(mu_can[i],theta_new[i].reshape(components_to_test,-1))
                print('pred_0',pred_0.shape)
                pred[i] = pred_0 
                covm = model_opt.compute_cov()
                # print('pis',output_t[0].cpu().detach().numpy())

                for k in range(components_to_test):
                    # print('pis',output_t[0].cpu().detach().numpy()[k])
                    # cov2[i]+=output_t[0].cpu().detach().numpy()[:,k]*covm[k].cpu().detach().numpy()
                    cov2[i]+=mu_can[i][k]*covm[k].cpu().detach().numpy()

                cov2[i]-=np.matmul(pred_0,pred_0.T)
                cov1[i]=np.linalg.det(cov2[i])
            print('predshape',pred.shape)
            cov = np.matmul(pred,pred.T)
            np.save('./EMLC/EDR/main_res/'+fname+'cov_can.npy',cov)

            print('covshape0',cov.shape)
            probs = np.mean(b2_can/(v2_can*(a2_can-1)),axis=1)+(cov1)
            np.save('./EMLC/EDR/main_res/'+fname+'var.npy',probs)

            ind_add = list(np.array(candidate_index)[list(np.argsort(probs)[::-1][:100])])
            train_index = train_index+ind_add
            print(ind_add)
            for ind_a in ind_add:
                candidate_index.remove(ind_a)
        np.save('./EMLC/EDR/main_res/'+fname+'loss_all.npy',loss_all)


if __name__ == "__main__":
    main()    