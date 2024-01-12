import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

import numpy as np

sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
sys.path.append('/'.join(os.getcwd().split('/')))
print(sys.path)

from saveHelpers.saveScript import save_losses
from Model.model import NNEvidential
# from DataLoader.dataRead import get_data
from DataLoader.dataRead import get_data_delicious
import matplotlib.pyplot as plt
device = torch.device('cuda:0')

save_name = "Evid_model_12_1024_abs_coef1.0_nosch"
if not os.path.exists(save_name):
    os.mkdir(save_name)
save_file = save_name + "/tr_trends.csv"

from trainingHelpers.lossFunctions import calculate_evidential_loss_constraints

def train(train_loader, model,  optimizer, epoch):
    loss_total = 0
    hyp_dict = {'alpha':0.0, 'beta':0.0, 'v':0.0, 'gamma':0.0}
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("batch_idx: ", batch_idx)
        ip, target = data.to(device), target.to(device)
        gamma, v, alpha, beta = model(ip)
        
        hyp_dict['alpha'] += torch.mean(alpha)
        hyp_dict['beta'] += torch.mean(beta)
        hyp_dict['gamma'] += torch.mean(gamma)
        hyp_dict['v'] += torch.mean(v)

        loss =  calculate_evidential_loss_constraints(batch_idx, target, gamma, v, alpha,beta,lambda_coef=1.0)
##########TODO
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.detach().cpu().item()

    for k, v in hyp_dict.items():
        hyp_dict[k] = v.detach().cpu().item()/(batch_idx + 1)

    return model, loss_total/(batch_idx+1), hyp_dict

def test(test_loader, model, criterion):
    model.zero_grad()
    loss_total = 0

    hyp_dict = {'alpha':0.0, 'beta':0.0, 'v':0.0, 'gamma':0.0}

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            ip, target = data.to(device), target.to(device)
            outputs, v, alpha, beta = model(ip)
            loss = criterion(outputs, target)
            loss_total += loss.detach().cpu().item()

            hyp_dict['alpha'] += torch.mean(alpha)
            hyp_dict['beta'] += torch.mean(beta)
            hyp_dict['gamma'] += torch.mean(outputs)
            hyp_dict['v'] += torch.mean(v)
    
    for k, v in hyp_dict.items():
        hyp_dict[k] = v.detach().cpu().item()/(batch_idx + 1)
    return loss_total/(batch_idx+1), hyp_dict

def test_unc(test_loader, model):
    '''
    Not used now. May enable later.
    '''
    model.zero_grad()

    mse_all, epis_all = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            ip, target = data.to(device), target.to(device)
            outputs, v, alpha, beta = model(ip)
            epis = v / (alpha - 1)/ beta

            error = [np.mean(x.detach().cpu().numpy()**2) for x in outputs - target]
            mse_all+=error

            epis = torch.mean(epis, dim = -1)

            epis_all += list(epis.detach().cpu().numpy())
    
    plt.scatter(epis_all, mse_all)
    plt.xlabel("Epistemic")
    plt.ylabel("MSE")
    plt.show()

def main():
    ip_dim, op_dim = 223, 6
    num_epochs = 4900

    model = NNEvidential(ip_dim, op_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2000,8000], gamma=0.1  # Decay by 10x at epochs 30 and 60
        )
    # Training data
    train_loader, test_loader = get_data_delicious()
    
    tr_loss_all = []
    test_loss_all = []

    hyp_dict_tr_all = {'alpha': [], 'beta':[], 'v':[], 'gamma':[]}
    hyp_dict_test_all = {'alpha': [], 'beta':[], 'v':[], 'gamma':[]}


    for epoch in range(num_epochs):
        model, tr_tr_loss, hyp_dict = train(train_loader, model, optimizer, epoch)
        tr_loss, hyp_dict_tr = test(train_loader, model, criterion)
        test_loss, hyp_dict_test = test(test_loader, model, criterion)
        tr_loss_all.append(tr_loss)
        test_loss_all.append(test_loss)
        # scheduler.step()

        # print("lr :", optimizer.param_groups[0]['lr'])
        print("Epoch: ", epoch, 
              "TR: ", np.round(tr_tr_loss, 5), 
              " Train Loss: ", np.round(tr_loss,5), 
              " Test Loss: ", np.round(test_loss,5))
        # print("Hyp: ", hyp_dict)
        to_save = [tr_tr_loss, tr_loss]
        for k, v in hyp_dict_tr.items():
            hyp_dict_tr_all[k] = hyp_dict_tr_all[k] + [hyp_dict_tr[k]]
            to_save.append(hyp_dict_tr[k])
        
        
        for k, v in hyp_dict_test.items():
            hyp_dict_test_all[k] = hyp_dict_test_all[k] + [hyp_dict_test[k]]
            to_save.append(hyp_dict_test[k])

        
        to_save.append(test_loss)
        if epoch in [4000,8000]:
            torch.save(model, f"model_new_{epoch}.pth")
        
        if epoch == 0:
            headers = ['Tr_Evid_Loss', 'Tr_MSE'] + [x+'tr' for x in hyp_dict_tr_all.keys()]
            headers += [x+'test' for x in hyp_dict_test_all.keys()] + ['Test_MSE']
            
            save_losses(save_file, header=headers, values=to_save)
        else:
            save_losses(save_file,values=to_save)


    st = 100
    plt.plot(tr_loss_all[st:], label = "Training Loss (MSE)")
    plt.plot(test_loss_all[st:], label = "Test Loss (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.savefig("simple_mse_evid_2.png")
    plt.show()

    for k, v in hyp_dict_tr_all.items():
        plt.cla()
        plt.plot(v, label = k + "Train")
        plt.plot(hyp_dict_test_all[k], label = "Test")
        plt.xlabel("Epochs")
        plt.ylabel("Hyperparameter")
        plt.legend()
        plt.savefig(f"{k}_plot_evid_2.png")
        plt.show()

    # test_unc(test_loader, model)
    # test_unc(train_loader, model)


if __name__ == "__main__":
    main()