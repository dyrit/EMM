import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AEModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AEModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 640),
            nn.ReLU(),
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128,output_dim),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 640),
            nn.ReLU(),
            nn.Linear(640,input_dim),
            # nn.ReLU(),
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return enc, out


class NNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NNModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,output_dim),
        )
    
    def forward(self, x):
        return self.fc(x)

class NNModel_AE_Based(nn.Module):
    def __init__(self, input_dim, output_dim, ae_model):
        super(NNModel_AE_Based, self).__init__()
        self.ae_model = ae_model
        self.fc = nn.Sequential(
            # nn.Linear(64,16),
            # nn.ReLU(),
            nn.Linear(64,output_dim),
            # nn.ReLU(),
        )
    
    def forward(self, x):
        # print(self.ae_model)
        with torch.no_grad():
            emb,_ = self.ae_model(x)
        return self.fc(emb)

class NNGaussian(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NNGaussian, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2*output_dim)
        )
        self.op_dim = output_dim
    
    def forward(self, x):
        op = self.fc(x)
        
        mu = op[:, :self.op_dim]
        log_sigma = op[:, self.op_dim:]
        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)

        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma

class NNEvidential(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NNEvidential, self).__init__()
        emb_size = 1024
        self.fc = nn.Sequential(
            nn.Linear(input_dim, emb_size),
            nn.ReLU(),
            # nn.Dropout(),

        )
        further_emb = 64
        self.transform_gamma = nn.Sequential(
            nn.Linear(emb_size, further_emb),
            nn.ReLU(),
            nn.Linear(further_emb, output_dim),
        )
        self.transform_v = nn.Sequential(
            nn.Linear(emb_size, further_emb),
            nn.ReLU(),
            nn.Linear(further_emb, output_dim),
        )
        self.transform_alpha = nn.Sequential(
            nn.Linear(emb_size, further_emb),
            nn.ReLU(),
            nn.Linear(further_emb, output_dim),
        )
        self.transform_beta = nn.Sequential(
            nn.Linear(emb_size, further_emb),
            nn.ReLU(),
            nn.Linear(further_emb, output_dim),
        )
        self.op_dim = output_dim
    
    def forward(self, x):
        emb = self.fc(x)
        
        gamma = self.transform_gamma(emb)
        logv = self.transform_v(emb)
        logalpha = self.transform_alpha(emb)
        logbeta = self.transform_beta(emb)

        gamma = F.softplus(gamma)
        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta =  0.1 + F.softplus(logbeta)

        return gamma, v, alpha, beta