import torch
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def deterministic(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_res(model_opt, x_test, y_test, mu, fname,num_classes,
              ground_truth, num_l,wnew,n=0,\
                alpha_test=None, alpha_train= None,\
                train_index=None,test_index=None,\
                handle='train_test',pi=None,bs_pred = None,ysum=False,\
                xtest=None,ytest=None,x_train=None,y_train=None):
    mse1 = 0
    mse4 = 0
    components_to_test = num_classes
    mu2 = np.array(mu)

    if (alpha_test is None):
        alpha_test = ground_truth[test_index]
        alpha_train = ground_truth[train_index]

    if(xtest is None):
        xtest = torch.tensor(x_test)
        ytest = torch.tensor(y_test)
        if(handle=='train_test'): 
            xtrain = torch.tensor(x_train)
            ytrain = torch.tensor(y_train)
    
    al1 = np.array(alpha_test)
    output_t = model_opt(xtest.float().to(device))

    al2 = np.array(output_t[0].cpu().detach().numpy())
    np.save('./EMLC/EDR/main_res/'+fname+'test_alpha_1.npy',al1)
    np.save('./EMLC/EDR/main_res/'+fname+'test_alpha_2.npy',al2)

    if(handle=='train_test'):
        np.save('./EMLC/EDR/main_res/'+fname+'x_train.npy',x_train)
        np.save('./EMLC/EDR/main_res/'+fname+'y_train.npy',y_train)

    np.save('./EMLC/EDR/main_res/'+fname+'x_test.npy',x_test)
    np.save('./EMLC/EDR/main_res/'+fname+'y_test.npy',y_test)

    #al2 = al2+1
    # al1 = al1.T
    pred1 = np.matmul(al1,mu2)
    al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
    pred2 = np.matmul(al2p,mu2)
    # y1 = np.array(ytest)
    y1=y_test
    mse2 = mean_squared_error(al1,al2)
    mse22 = mean_squared_error(alpha_test,al2)

    # if ysum:
    #     y1s = np.sum(y1,0)
    #     col = np.where(y1s!=0)[0]
    #     # print('sum',y1s[col])
    #     y2 = y1[:,col]
    #     pred12 = pred1[:,col]
    #     pred22 = pred2[:,col]
    #     mse5 = mean_squared_error(al1,alpha_test)
    # else:
    #     y2=y1
    #     pred12 = pred1
    #     pred22 = pred2




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



    test_auc_micro = metrics.roc_auc_score(y2,pred12,average='micro')
    test_auc_macro = metrics.roc_auc_score(y2,pred12,average='macro')
    print('test mi:alpha1',test_auc_micro)
    print('test ma:alpha1',test_auc_macro)

    test_auc_micro_2 = metrics.roc_auc_score(y2,pred22,average='micro')
    test_auc_macro_2 = metrics.roc_auc_score(y2,pred22,average='macro')   
    print('test mi:alpha2',test_auc_micro_2)
    print('test ma:alpha2',test_auc_macro_2)

    if (bs_pred is not None):
        al_train = bs_pred[train_index]
        al_test = bs_pred[test_index]
        pred3 = np.matmul(al_test,mu2)
        pred32 = pred3[:,col]
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


# training results 
    #al2 = al2+1
    al1 = np.array(alpha_train)
    output_t = model_opt(xtrain.float().to(device))

    al2 = np.array(output_t[0].cpu().detach().numpy())
    # al1 = al1.T
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


    pred1 = np.matmul(al1,mu2)
    al2p = al2/np.repeat(np.sum(al2,axis=1).reshape(-1,1),num_classes,axis=1)
    pred2 = np.matmul(al2p,mu2)

    # y1 = np.array(ytrain)
    y1=y_train
    y1s = np.sum(y1,0)
    col = np.where(y1s!=0)[0]
    y2 = y1[:,col]
    pred12 = pred1[:,col]
    pred22 = pred2[:,col]

    auc_micro = metrics.roc_auc_score(y2,pred12,average='micro')
    auc_macro = metrics.roc_auc_score(y2,pred12,average='macro')
    print('mi:alpha1',auc_micro)
    print('ma:alpha1',auc_macro)

    auc_micro_2 = metrics.roc_auc_score(y2,pred22,average='micro')
    auc_macro_2 = metrics.roc_auc_score(y2,pred22,average='macro')   
    print('mi:alpha2',auc_micro_2)
    print('ma:alpha2',auc_macro_2)

    if (bs_pred is not None):
        pred3 = np.matmul(al_train,mu2)
        pred32 = pred3[:,col]

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
    # print("mse5",mse5)
    # print(fname)
    # print('y1',ytest[:10,:])
    # print('yt',y_test[:10])

    if (bs_pred is not None):
        datapoint = [n, mse1,mse2,mse3,mse4,auc_macro,auc_macro_2,auc_macro_3,test_auc_macro,test_auc_macro_2,test_auc_macro_3,auc_macro10,\
                     test_auc_micro,test_auc_micro_2,test_auc_micro_3,auc_micro10]
    else:
        datapoint = [n,mse2,mse3,auc_macro,auc_macro_2,test_auc_macro,test_auc_macro_2,auc_macro10,test_auc_micro,test_auc_micro_2,auc_micro10]


    return datapoint