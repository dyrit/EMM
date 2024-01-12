import torch
import numpy as np

def NIG_NLL(it, y, mu, v, alpha, beta):
    epsilon = 1e-16
    twoBlambda = 2*beta*(1+v)

    a1 = 0.5723649429247001 - 0.5 * torch.log(v+epsilon)
    a2a = - alpha*torch.log( 2*beta +epsilon)
    a2b = - alpha * torch.log(1 + v)
    a3 = (alpha+0.5) * torch.log( v*(y-mu)**2 + twoBlambda + epsilon)
    a4 = torch.lgamma(alpha) - torch.lgamma(alpha+0.5)

    a2 = a2a + a2b

    nll = a1 + a2 + a3 + a4


    return nll

def NIG_Reg(y, mu, v, alpha, beta):
    
    # error = torch.abs(y-mu)#**2
    error = (y-mu)**2
    # error = error.detach()
    

    evi = v + alpha + 1/(beta+1e-15)
    reg = error*evi

    return reg


def calculate_evidential_loss(it, y, mu, v, alpha, beta, lambda_coef=1.0):

    nig_nllt = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_reg =NIG_Reg(y, mu, v, alpha,beta)

    ev_sum = nig_nll  + lambda_coef*nig_reg
    evidential_loss = torch.mean(ev_sum)

    return evidential_loss

def calculate_evidential_loss_constraints(it, y, mu, v, alpha, beta, lambda_coef=1.0):

    nig_nll = NIG_NLL(it, y, mu, v, alpha,beta)
    nig_reg =NIG_Reg(y, mu, v, alpha,beta)

    ev_sum = nig_nll  + lambda_coef*nig_reg
    evidential_loss = torch.mean(ev_sum)

    return evidential_loss
