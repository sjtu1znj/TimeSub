import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir))
import torch
import numpy as np
from lifelines import KaplanMeierFitter  
import torch.nn as nn
import random
from torch.autograd import Variable
from prediction import prediction_ability, t_averaged_RE, t_median_RE
import time
import copy
import math
learning_B = 5e-2
num_B = 50
device="cuda" if torch.cuda.is_available() else "cpu"
def untie_tensor_with_preprocessing(
    T_B: torch.Tensor,
    tolerance: float = 1e-6
) -> torch.Tensor:
    """
    Pre-processes and unties a PyTorch tensor containing tied or very close time points.

    This function operates in two main steps:
    1. Pre-processing:
       a. Clamps all time points smaller than `tolerance` to `tolerance`.
       b. Iterates through the sorted time points. If a time point is closer to the
          previous one than `tolerance`, it's "snapped" to the previous value. This
          is done sequentially to handle chains of close values correctly.
    2. Untying:
       a. For the new ties created during pre-processing, a small, increasing
          perturbation is added to break them.
       b. The perturbation unit, epsilon, is calculated as `0.9 * tolerance / n1`,
          where n1 is the size of the largest tie group.

    Args:
        T_B (torch.Tensor): The input tensor, with shape (n, 1) and dtype torch.float32.
        tolerance (float): The threshold for snapping close values and setting the minimum time.

    Returns:
        torch.Tensor: The processed tensor with the same shape and dtype as the input.
    """
    # Ensure the input is float32, as the logic is sensitive to precision.
    if T_B.dtype != torch.float32:
        raise TypeError(f"Input tensor T_B must be torch.float32, but got {T_B.dtype}")
        
    if T_B.numel() <= 1:
        # For a single element or an empty tensor, just apply the minimum time threshold.
        return torch.clamp(T_B.clone(), min=tolerance)

    # --- Preparation ---
    T_flat = T_B.squeeze()
    device = T_flat.device

    # --- Step 1: Sort the tensor ---
    # Sorting is crucial for identifying ties and handling sequential snapping.
    sorted_T, sort_indices = torch.sort(T_flat)

    # --- Step 2: Pre-processing - Snap close values together ---
    # A loop is necessary here because the snapping of each value depends on the
    # *processed* value of the previous one. This correctly handles chains like [t, t+0.6e-6, t+1.2e-6].
    preprocessed_T = sorted_T.clone()
    for i in range(1, len(preprocessed_T)):
        if preprocessed_T[i] - preprocessed_T[i-1] < tolerance:
            preprocessed_T[i] = preprocessed_T[i-1]
            
    # --- Step 3: Pre-processing - Enforce minimum time ---
    # This is done efficiently using torch.clamp after snapping.
    preprocessed_T = torch.clamp(preprocessed_T, min=tolerance)

    # --- Step 4: Calculate Epsilon ---
    # Calculate tie statistics on the newly pre-processed data.
    unique_vals, counts = torch.unique(preprocessed_T, return_counts=True)
    
    if counts.numel() == 0 or counts.max() <= 1:
        # If there are no ties (max count is 1), no perturbation is needed.
        epsilon = 0.0
    else:
        max_ties = counts.max().item()
        # Calculate epsilon according to the specified formula.
        epsilon = (0.9 * tolerance) / max_ties
    
    # If no perturbation is needed, we can restore the order and return directly.
    if epsilon == 0.0:
        untied_flat = preprocessed_T[sort_indices.argsort()]
        return untied_flat.view_as(T_B)

    # --- Step 5: Calculate tie ranks ---
    # This vectorized method efficiently finds the rank of each element within its tie group.
    _, inverse_indices = torch.unique_consecutive(preprocessed_T, return_inverse=True)
    perm = torch.arange(len(preprocessed_T), device=device)
    group_starts_mask = torch.cat([torch.tensor([True], device=device), inverse_indices[1:] != inverse_indices[:-1]])
    group_start_indices = perm[group_starts_mask]
    tie_ranks = perm - group_start_indices[inverse_indices]

    # --- Step 6: Apply the perturbations ---
    perturbations = tie_ranks.to(torch.float32) * epsilon
    untied_sorted_T = preprocessed_T + perturbations

    # --- Step 7: Restore the original order ---
    # Use argsort of the original sort indices to create an "unsorting" permutation.
    untied_flat = untied_sorted_T[sort_indices.argsort()]

    return untied_flat.view_as(T_B)
def set_seed(seed):
    # This function sets the random seed for various libraries to ensure reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(0)
class BaseNet(nn.Module):
    # Base class for the neural network model.
    def __init__(self, input_size, hidden_size, depth, output_size=1, activation=nn.ReLU):
        super(BaseNet, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation())

        # Hidden layers
        for i in range(1, depth):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a Sequential module
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

def create_net_class(input_size, hidden_size, depth, output_size=1, activation=nn.ReLU):
    # Factory function to create a custom neural network class.
    class CustomNet(BaseNet):
        def __init__(self):
            super(CustomNet, self).__init__(input_size,  hidden_size, depth, output_size, activation)
        def detach(self):
            for param in self.parameters(): 
                param.detach_()
            return self
    return CustomNet
class cr_data:
    # competing risk data.
    def __init__(self,T,Delta,epsilon,Z,TX=None,cov=None,time_breaks=None):
        # T: survival time, Delta: event indicator, epsilon: cause of event, Z: covariates
        self.T = T
        self.Delta = Delta
        self.epsilon = epsilon
        self.Z = Z
        self.n = len(T)
        self.Ghat = None
        self.TX = TX
        self.cov = cov
        self.time_breaks = time_breaks
        self.R = None
        self.cov_dim = Z.shape[1]
        self.models = {'nn_tv':None,'nn_ti':None,'spline':None,'linear':None}
        self.test_models = {'nn_tv':None,'nn_ti':None}
        self.model_seeds = {'nn_tv':None,'nn_ti':None}
        self.dLam = None
        self.expg = None
        self.sort_indices = None
        self.sorted_times = None
        self.unique_indices = None
        self.model_ti = None
        self.model_tv = None
        self.plr_st = None
        self.plr_st_B = []
        self.plr_si = None
        self.plr_si_B = []
        self.data1 = None
    def __getitem__(self, index):
        # return a new cr_data instance containing the data at index index
        if self.TX is not None:
            return cr_data(
                T=self.T[index],
                Delta=self.Delta[index],
                epsilon=self.epsilon[index],
                Z=self.Z[index],
                TX=self.TX[index][:,index,:]
            )
        return cr_data(
            T=self.T[index],
            Delta=self.Delta[index],
            epsilon=self.epsilon[index],
            Z=self.Z[index]
        )
    def G_hat(self):
        # G_hat is the estimated cumulative incidence function for the censored event.
        device1 = self.T.device
        self.to("cpu")
        T = self.T
        delta = self.Delta
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=1-delta)
        km = kmf.survival_function_["KM_estimate"]
        Ghat = torch.tensor(km.loc[T[:,0]].values.reshape(len(T[:,0]),1))
        self.Ghat = Ghat
        self.to(device1)
    
    def Riskmatrix(self):
        # Risk matrix is a matrix that indicates the risk set at each time point.
        if self.Ghat == None:
            self.G_hat()
        device1 = self.T.device
        T = self.T.to(device1)
        Ghat = self.Ghat.to(device1)
        Delta = self.Delta.to(device1)
        epsilon = self.epsilon.to(device1)
        n = len(T)
        G = Ghat/Ghat.T
        G[torch.isnan(G)] = 0
        G[torch.isinf(G)] =-0
        R1 = T <= T.T
        R2 = torch.minimum(T >= T.T,torch.ones((n,1)).to(device1)@(epsilon-1).T)
        R2 = torch.minimum(R2,torch.ones((n,1)).to(device1)@(Delta).T)*G
        R = torch.maximum(R1,R2)
        self.R = R

    
    def X2TX(self):
        # TX is a tensor that combines the survival time and covariates.
        if self.TX == None:
            if len((self.Z).shape) == 2:
                d = self.cov_dim
                n = self.n
                T_test = self.T.to(device)
                X_test = self.Z.to(device)
                TX_test = torch.zeros((n, n, d+1)).to(device)
                TX_test[:, :, 0] = T_test
                for i in range(d):
                    TX_test[:,:,1+i] = X_test[:,i].reshape((1,n))
                self.TX = TX_test
            if len((self.Z).shape) == 3:
                TX = torch.zeros((self.Z.shape[0], self.Z.shape[1], self.Z.shape[2]+1))
                TX[:,:,0] = self.T.to(device)
                TX[:, :, 1:] = self.Z.to(device)
                self.TX = TX
    def to(self,device):
        # Move the data to the specified device (CPU or GPU).
        self.T = self.T.to(device)
        self.Delta = self.Delta.to(device)
        self.epsilon = self.epsilon.to(device)
        self.Z = self.Z.to(device)
        if self.Ghat != None:
            self.Ghat = self.Ghat.to(device)
        if self.TX != None:
            self.TX = self.TX.to(device)
        if self.R != None:
            self.R = self.R.to(device)
        
    def check_R(self):
        # Check if the risk matrix R is not None.
        assert self.R is not None, "The value of self.R cannot be None"
    
    def check_TX(self):
        # Check if the TX tensor is not None.
        assert self.TX is not None, "The value of self.TX cannot be None"

    def check_Ghat(self):
        # Check if the Ghat tensor is not None.
        assert self.Ghat is not None, "The value of self.Ghat cannot be None"
    
    def data_process(self):
        # Process the data by computing G_hat, Riskmatrix, and X2TX.
        # This function is called to prepare the data for training or evaluation.
        self.G_hat()
        self.Riskmatrix()
        self.X2TX()
        self.to(device)
    def remove_column(self,column_index):
        
        # Remove the specified column from the Z tensor.
        newcov = torch.cat((self.cov[...,:column_index],self.cov[...,column_index+1:]),axis=-1)
        newZ = torch.cat((self.Z[...,:column_index],self.Z[...,column_index+1:]),axis=-1)
        if self.TX is not None:
            newTX = torch.cat((self.TX[...,:column_index+1],self.TX[...,column_index+2:]),axis=-1)
            return cr_data(
                T=self.T,
                Delta=self.Delta,
                epsilon=self.epsilon,
                Z=newZ,
                TX = newTX,
                cov = newcov,
                time_breaks = self.time_breaks
            )
        else:
            return cr_data(
                T=self.T,
                Delta=self.Delta,
                epsilon=self.epsilon,
                Z=newZ,
                cov = newcov,
                time_breaks = self.time_breaks
            )
    def to_torch(self):
        # Convert all numpy arrays to torch tensors and preprocess T to handle ties.
        self.T = untie_tensor_with_preprocessing(torch.from_numpy(self.T.astype(np.float32)))
        self.Delta = torch.from_numpy(self.Delta.astype(np.float32))
        self.epsilon = torch.from_numpy(self.epsilon.astype(np.float32))
        self.Z = torch.from_numpy(self.Z.astype(np.float32))
        if self.TX is not None:
            self.TX = torch.from_numpy(self.TX.astype(np.float32))
        self.cov = torch.from_numpy(self.cov.astype(np.float32))
    def nn_tv(self, 
              Nets = [create_net_class(6, 10, 1),create_net_class(6, 20, 3)],
              batch_num = 1,
              learning_rate1=[1e-2,1e-3],
              random_num = 5,
             best_model_seed_output = False,
             num_epoch = 300):
        # time-varying nonparametric subdistribution hazard model (TNSHM) using neural networks.
        # Nets: list of neural network classes to be used. (6,10,1) means input size 6, hidden size 10, depth 1. 
        # batch_num: number of batches for training.
        # learning_rate1: list of learning rates for different stages of training. Corresponds to Nets.
        # random_num: number of random initializations for model selection.
        # num_epoch: number of epochs for training.
        start_time = time.time()
        if best_model_seed_output:
            model,best_model_seed,best_epoch = estimation(self,batch_num,Nets,random_num=random_num,learning_rate1=learning_rate1,best_model_seed_output = best_model_seed_output,num_epoch=num_epoch)
            self.model_seeds['nn_tv'] = (best_model_seed,best_epoch)
            num_valid = np.minimum(1000,int(0.25*self.n))
            batches,data_valid = train_valid(self, num_valid, batch_num)
            # if best_epoch > 100:
            #     test_model = train_DP(best_model_seed, Nets[0], MyLoss, best_epoch-100, learning_rate1, batches, data_valid, True)[1]
            #     self.test_models['nn_tv'] = test_model
        else:
            model = estimation(self,batch_num,Nets,random_num=random_num,learning_rate1=learning_rate1,best_model_seed_output = best_model_seed_output)
        model.eval()
        self.to('cpu')
        def model1(data_test):
            device1 = data_test.T.device
            data_test.to('cpu')
            model.to('cpu')
            if data_test.TX is None:
                data_test.X2TX()
            data_test.to('cpu')
            data_test.TX = data_test.TX.to('cpu')
            output = model(data_test.TX)[:,:,0].to('cpu')
            data_test.to(device1)
            return output
        end_time = time.time()
        self.models['nn_tv'] = (model1,end_time - start_time)
        self.model_tv = model
    
    def spline(self,
               num_spline = 10,
               num_epoch = 500,
               learning_rate = 0.01):
        # the spline-based time-varying coeffcient subdistribution hazard model (TSHM) using cubic splines.
        # num_spline: number of spline basis functions.
        # num_epoch: number of epochs for training.
        # learning_rate: learning rate for optimization.
        start_time = time.time()
        self.data_process()
        d = self.Z.shape[self.Z.dim()-1]
        n = self.Z.shape[0]
        m = self.T.max()
        B = cubic_spline(num_spline,self.T[:,0],m).float()
        Gamma = torch.zeros((num_spline+4,d), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([Gamma], lr=learning_rate)
        def gv(BG,Z):
            if Z.dim() == 3:
                return ((BG).reshape(n,1,d)*Z).sum(axis=2)
            elif Z.dim() == 2:
                return BG@Z.T
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            g_value = gv(B@Gamma,self.Z)
            l = -patial_likelihood(self, g_value)
            l.backward()
            optimizer.step()
        self.to('cpu')
        Gamma = Gamma.cpu().detach()
        def model(data_test):
            device1 = data_test.T.device
            data_test.to('cpu')
            B_test = cubic_spline(num_spline, data_test.T[:,0], m).float().cpu()
            output = gv(B_test@Gamma, data_test.Z)
            data_test.to(device1)
            return output
        end_time = time.time()
        self.models['spline'] = (model, end_time - start_time)
        
    
    def linear(self,
               num_epoch = 200,
               learning_rate = 0.01):
        # the original subdistribution hazard model (SHM) 
        # num_epoch: number of epochs for training.
        # learning_rate: learning rate for optimization.
        start_time = time.time()
        self.data_process()
        d = self.Z.shape[self.Z.dim()-1]
        n = self.Z.shape[0]
        beta = torch.zeros((1,)*(self.Z.dim()-1)+(d,),requires_grad=True,device=device)
        optimizer = torch.optim.Adam([beta], lr=learning_rate)
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            g_value = (beta*self.Z).sum(axis=self.Z.dim()-1) + torch.zeros((n,n)).to(device)
            l = -patial_likelihood(self, g_value)
            l.backward()
            optimizer.step()
        beta = beta.cpu().detach()
        def model(data_test):
            device1 = data_test.T.device
            data_test.to('cpu')
            output = (beta*data_test.Z).sum(axis=self.Z.dim()-1) + torch.zeros((data_test.n,data_test.n))
            data_test.to(device1)
            return output
        self.to('cpu')
        end_time = time.time()
        self.models['linear'] = (model, end_time - start_time)
    
    def nn_ti(self,
            Nets = [create_net_class(5, 10, 1),create_net_class(5, 20, 3)],
            batch_num = 1,
            learning_rate1=[1e-2,1e-3],
            random_num = 5,
            best_model_seed_output = False,
            num_epoch = 300):
        # the nonparametric subdistribution hazard model (NSHM) using neural networks.
        # Nets: list of neural network classes to be used. 
        # batch_num: number of batches for training.
        # learning_rate1: list of learning rates for different stages of training. Corresponds to Nets.
        # random_num: number of random initializations for model selection.
        # num_epoch: number of epochs for training.
        start_time = time.time()
        if self.TX is None:
            data_TI = cr_data(self.T, self.Delta, self.epsilon, self.Z)
        else:
            data_TI = cr_data(self.T, self.Delta, self.epsilon, self.Z, self.TX[:,:,1:])
        if self.Z.dim() == 2:
            if best_model_seed_output:
                model,best_model_seed,best_epoch = estimation(data_TI,batch_num,Nets,random_num=random_num,time_varying=False,learning_rate1=learning_rate1,best_model_seed_output = best_model_seed_output,num_epoch=num_epoch)
                self.model_seeds['nn_ti'] = (best_model_seed,best_epoch)
                num_valid = np.minimum(1000,int(0.25*self.n))
                batches,data_valid = train_valid(data_TI, num_valid, batch_num)
                # if best_epoch > 100:
                #     test_model = train_DP(best_model_seed, Nets[0], MyLoss, best_epoch-100, learning_rate1, batches, data_valid, False)[1]
                    # self.test_models['nn_ti'] = test_model
            else:
                model = estimation(data_TI,batch_num,Nets,random_num=random_num,time_varying=False,learning_rate1=learning_rate1)
        elif self.Z.dim() == 3:
            if best_model_seed_output:
                model,best_model_seed,best_epoch = estimation(data_TI,batch_num,Nets,random_num=random_num,learning_rate1=learning_rate1,best_model_seed_output = best_model_seed_output)
                self.model_seeds['nn_ti'] = (best_model_seed,best_epoch)
                num_valid = np.minimum(1000,int(0.25*self.n))
                batches,data_valid = train_valid(data_TI, num_valid, batch_num)
                if best_epoch > 100:
                    test_model = train_DP(best_model_seed, Nets[0], MyLoss, 100, learning_rate1, batches, data_valid, True)[1]
                    self.test_models['nn_ti'] = test_model
            else:
                model = estimation(data_TI,batch_num,Nets,random_num=random_num,learning_rate1=learning_rate1)
        model.eval()
        def model1(data_test):
            device1 = data_test.T.device
            data_test.to('cpu')
            model.to('cpu')
            if data_test.TX is None:
                data_test.X2TX()
            data_test.to('cpu')
            data_test.TX = data_test.TX.to('cpu')
            output = model(data_test.TX[:,:,1:])[:,:,0]
            data_test.to(device1)
            return output
        self.to('cpu')
        end_time = time.time()
        self.models['nn_ti'] = (model1, end_time - start_time)
        self.model_ti = model
    def run_allmodels(self,tau, data_test, g0_torch,batch_num=1):
        self.nn_tv(batch_num=batch_num)
        self.spline()
        self.linear()
        self.nn_ti(batch_num=batch_num) 
        self.summary(tau, data_test, g0_torch)       
    def summary(self,tau,data_test,g0_torch):
        output = '-' * 50 + '\n'
        for key in self.models:
            if self.models[key] is not None:
                output += f"Model: {key}:\n"
                model, model_time = self.models[key]
                g_torch_train = model(self)
                g_torch = model(data_test)
                A_h = Lam0hat(self,g_torch_train)
                Ctau, y = prediction_ability(data_test,g_torch,g0_torch)
                q1 = y[int(0.25 * len(y))]
                q2 = y[int(0.5 * len(y))]
                q3 = y[int(0.75 * len(y))]
                output += f"A1_h: {A_h[torch.where(self.T ==self.T[self.T<tau/4].max())[0]].item()}\n"
                output += f"A2_h: {A_h[torch.where(self.T ==self.T[self.T<tau/2].max())[0]].item()}\n"
                output += f"A3_h: {A_h[torch.where(self.T ==self.T[self.T<tau*3/4].max())[0]].item()}\n"
                output += f"t_averaged_RE: {t_averaged_RE(g_torch, g0_torch).item()}\n"
                output += f"t_median_RE: {t_median_RE(g_torch, g0_torch).item()}\n"
                output += f"Ctau: {Ctau}\n"
                output += f"q1: {q1}, q2: {q2}, q3: {q3}\n"
                output += f"Time taken for {key}: {model_time:.2f} seconds\n"
                output += '-' * 50 + '\n'
        self.output = output
    def Structure_data_B_pre(self,Nets = [create_net_class(5, 12, 3)],learning_rate = 1e-3,num_epoch = 300):
        if self.R == None:
            self.Riskmatrix()
        if self.models['nn_ti'] == None:
            self.nn_ti(Nets = Nets
                       ,learning_rate1 = learning_rate,
                       num_epoch = num_epoch)
        device1 = self.T.device
        self.to("cpu")
        null_g = self.models['nn_ti'][0]
        Lam_hat = ((self.T <= (self.T.T))*(2-self.epsilon)*self.Delta/(torch.mean(self.R*torch.exp(null_g(self)),axis = 1).reshape(-1,1))).mean(axis = 0)

        sort_indices = self.T.sort(axis=0).indices[:, 0].cpu().numpy()
        sorted_times = self.T[sort_indices].cpu().numpy().flatten()
        sorted_Lam0_hat = Lam_hat[sort_indices].detach().cpu().numpy().flatten()
        unique_times, unique_indices = np.unique(sorted_times, return_index=True)
        unique_Lam0_hat = sorted_Lam0_hat[unique_indices]
        final_times = np.insert(unique_times, 0, 0)
        final_Lam0_hat = np.insert(unique_Lam0_hat, 0, 0)
        dLam = np.diff(final_Lam0_hat).reshape(-1, 1)
        expg = torch.exp(null_g(self))
        self.dLam = dLam
        self.expg = expg.detach()
        self.sort_indices = sort_indices
        self.sorted_times = sorted_times
        self.unique_indices = unique_indices
        self.to(device1)
    def Structure_data_B(self,boot_seed,Nets = [create_net_class(5, 12, 3)],learning_rate = 1e-3,num_epoch = 300):
        device1 = self.T.device
        self.to("cpu")
        if self.dLam is None:
            self.Structure_data_B_pre(Nets,learning_rate,num_epoch = num_epoch)
        n = self.n
        set_seed(boot_seed)
        index_B = np.random.randint(0,n,size=n)
        Lam_Z_B = torch.cumsum((self.expg[index_B][:,self.sort_indices][:,self.unique_indices])*self.dLam,dim=0).detach().numpy()
        U_star_B = np.zeros(n)
        E_B = np.random.exponential(scale=1.0, size=n)
        U_index_B = (Lam_Z_B < E_B).sum(axis=0)
        U_star_B[U_index_B >= n] = np.inf
        U_star_B[U_index_B<n] = self.sorted_times[U_index_B[U_index_B<n]]
        n2 = torch.sum((self.epsilon == 2)[:,0]).item()
        n2_B = (U_index_B>= n).sum()
        U2_B_index = np.random.randint(0,n2,size=n2_B)
        U2_dic = torch.where((self.epsilon == 2)[:,0])[0]
        U_star_B[U_index_B >= n] = self.T[U2_dic[U2_B_index]][:,0]
        u_C_B = torch.rand(n)
        C_B = self.sorted_times[torch.minimum(((1-self.Ghat[self.sort_indices])<u_C_B).sum(axis=0),torch.tensor(n-1))]
        T_B = torch.tensor(np.minimum(U_star_B, C_B)).float().reshape(-1, 1)
        Delta_B = torch.tensor(U_star_B <= C_B).float().reshape(-1, 1)
        epsilon_B = (torch.tensor(2-(U_index_B<n)).reshape(-1, 1)*(Delta_B)).float()
        Z_B = torch.zeros(self.Z.shape)
        cov_B = torch.zeros(self.cov.shape)
        cov_B[U_index_B<n] = self.cov[index_B][U_index_B<n]
        cov_B[U_index_B>=n] = self.cov[U2_dic[U2_B_index]]
        t_vec = T_B
        if len(self.Z.shape) == 3:
            time_breaks_tensor = torch.as_tensor(self.time_breaks, dtype=t_vec.dtype, device=t_vec.device)
            interval_indices = torch.bucketize(t_vec.squeeze(-1), time_breaks_tensor, right=True) - 1
            Z_B = cov_B[:,interval_indices,:].permute(1, 0, 2)
        if len(self.Z.shape) == 2:
            Z_B = cov_B
        T_B = untie_tensor_with_preprocessing(T_B)
        if len(self.Z.shape) == 2:
            n = self.Z.shape[0]
            d = self.Z.shape[1]
            TX_B = torch.zeros((n, n, d+1))
            TX_B[:,:,0] = T_B
            for i in range(d):
                TX_B[:, :, i+1] = Z_B[:, i].reshape(1, n)
        if len(self.Z.shape) == 3:
                TX_B = torch.zeros((self.Z.shape[0], self.Z.shape[1], self.Z.shape[2]+1))
                TX_B[:,:,0] = T_B
                TX_B[:, :, 1:] = Z_B
        data_B = cr_data(T_B, Delta_B, epsilon_B, Z_B, TX_B)
        self.to(device1)
        return data_B
    def structure_test(self,B_seeds,learning_rate = 1e-3, num_epoch = 300, num_epoch_ = num_B, num_epoch_B = num_B, learning_rate_B = learning_B, Nets = [create_net_class(6, 12, 3)],max_batch_size = 2000 ,valid_rate = 0.5,Nets_TI = [create_net_class(5, 12, 3)],log = None,num_model = 5, num_model_B = 5):
        # Perform structure test.
        # B_seeds: list of bootstrap seeds.
        # learning_rate: learning rate for initial model training.
        # num_epoch: number of epochs for initial model training.
        # num_epoch_: number of epochs for model training during test statistic calculation.
        # num_epoch_B: number of epochs for model training during bootstrap test statistic calculation.
        # learning_rate_B: learning rate for model training during test statistic calculation.
        # Nets: list of neural network classes for the main model.
        # max_batch_size: maximum batch size for training.
        # valid_rate: proportion of data used for validation.
        # Nets_TI: list of neural network classes for the null model.
        # log: file path for logging results.
        # num_model: number of random initializations for model training during test statistic calculation.
        # num_model_B: number of random initializations for model training during bootstrap test statistic calculation
        
        start_time = time.time()
        n = self.n
        batch_num = math.ceil(n/max_batch_size)
        if self.models['nn_ti'] == None:
            self.nn_ti(Nets = Nets_TI,learning_rate1=learning_rate,batch_num=batch_num,best_model_seed_output =True,num_epoch=num_epoch)
        if self.TX == None:
            data_TI = cr_data(self.T, self.Delta, self.epsilon, self.Z)
        else:
            data_TI = cr_data(self.T, self.Delta, self.epsilon, self.Z, self.TX[:,:,1:])
        num_valid = int(valid_rate*n)
        batches,data_valid = train_valid(self, num_valid, batch_num)
        batches_TI, data_valid_TI = train_valid(data_TI, num_valid, batch_num)
        loss1_list = []
        loss2_list = []
        for model_seed in range(num_model):
            if self.Z.dim() == 2:
                loss_valid = train_DP(model_seed, Nets_TI[0], MyLoss, num_epoch_, learning_rate_B, batches_TI, data_valid_TI, time_varying = False)[2]
            else: 
                loss_valid = train_DP(model_seed, Nets_TI[0], MyLoss, num_epoch_, learning_rate_B, batches_TI, data_valid_TI, time_varying = True)[2]
            loss1_list.append(loss_valid)
            loss_valid = train_DP(model_seed, Nets[0], MyLoss, num_epoch_, learning_rate_B, batches, data_valid, time_varying = True)[2]
            loss2_list.append(loss_valid)
        l1 = (np.array(loss1_list)[~np.isnan(np.array(loss1_list))]).min()
        l2 = (np.array(loss2_list)[~np.isnan(np.array(loss2_list))]).min()
        plr_st = 2*(l1-l2)
        end_time = time.time()
        self.plr_st = (plr_st,end_time-start_time)
        if log is not None:
            with open(log, 'a') as f:
                output = f"Structure test statistics = {plr_st}.\n"
                output+= f"Time taken for statistics = {end_time-start_time}.\n"
                f.write(output + '\n'+ '-'*50 + '\n')
                
        for B_seed in B_seeds:
            start_time = time.time()
            data_B = self.Structure_data_B(B_seed,Nets_TI,learning_rate)
            data_B_TI = cr_data(data_B.T, data_B.Delta, data_B.epsilon, data_B.Z, data_B.TX[:,:,1:])
            num_valid = int(valid_rate*n)
            batches_B,data_valid_B = train_valid(data_B, num_valid, batch_num)
            batches_B_TI, data_valid_B_TI = train_valid(data_B_TI, num_valid, batch_num)
            loss1_list = []
            loss2_list = []
            for model_seed in range(num_model_B):
                if self.Z.dim() == 2:
                    loss_valid = train_DP(model_seed, Nets_TI[0], MyLoss, num_epoch_B, learning_rate_B, batches_B_TI, data_valid_B_TI, time_varying = False)[2]
                else: 
                    loss_valid = train_DP(model_seed, Nets_TI[0], MyLoss, num_epoch_B, learning_rate_B, batches_B_TI, data_valid_B_TI, time_varying = True)[2]
                loss1_list.append(loss_valid)
                loss_valid = train_DP(model_seed, Nets[0], MyLoss, num_epoch_B, learning_rate_B, batches_B, data_valid_B, time_varying = True)[2]
                loss2_list.append(loss_valid)
            l1 = (np.array(loss1_list)[~np.isnan(np.array(loss1_list))]).min()
            l2 = (np.array(loss2_list)[~np.isnan(np.array(loss2_list))]).min()
            plr_st_B = 2*(l1-l2)
            end_time = time.time()
            self.plr_st_B.append((plr_st_B,end_time-start_time))
            if log is not None:
                with open(log, 'a') as f:
                    output = f"Bootstrap seed is {B_seed}.\n"
                    output += f"Structure test statistics = {plr_st_B}.\n"
                    output+= f"Time taken for Bootstrap statistics = {end_time-start_time}.\n"
                    f.write(output + '\n')
    def Significance_data_B_pre(self,Nets = [create_net_class(5, 12, 3)],learning_rate = 1e-3,num_epoch = 300):
        if self.R == None:
            self.Riskmatrix()
        if self.models['nn_tv'] == None:
            self.nn_tv(Nets = Nets
                       ,learning_rate1 = learning_rate,num_epoch=num_epoch)
        device1 = self.T.device
        self.to("cpu")
        null_g = self.models['nn_tv'][0]
        Lam_hat = ((self.T <= (self.T.T))*(2-self.epsilon)*self.Delta/(torch.mean(self.R*torch.exp(null_g(self)),axis = 1).reshape(-1,1))).mean(axis = 0)
        sort_indices = self.T.sort(axis=0).indices[:, 0].cpu().numpy()
        sorted_times = self.T[sort_indices].cpu().numpy().flatten()
        sorted_Lam0_hat = Lam_hat[sort_indices].detach().cpu().numpy().flatten()
        unique_times, unique_indices = np.unique(sorted_times, return_index=True)
        unique_Lam0_hat = sorted_Lam0_hat[unique_indices]
        final_times = np.insert(unique_times, 0, 0)
        final_Lam0_hat = np.insert(unique_Lam0_hat, 0, 0)
        dLam = np.diff(final_Lam0_hat).reshape(-1, 1)
        expg = torch.exp(null_g(self))
        self.dLam = dLam
        self.expg = expg.detach()
        self.sort_indices = sort_indices
        self.sorted_times = sorted_times
        self.unique_indices = unique_indices
        self.to(device1) 
    def Significance_data_B(self,data,boot_seed,Nets = [create_net_class(5, 12, 3)],learning_rate = 1e-3,num_epoch = 300):
        device1 = self.T.device
        self.to("cpu")
        if self.dLam is None:
            self.Significance_data_B_pre(Nets,learning_rate,num_epoch=num_epoch)
        n = self.n
        set_seed(boot_seed)
        index_B = np.random.randint(0,n,size=n)
        Lam_Z_B = torch.cumsum((self.expg[index_B][:,self.sort_indices][:,self.unique_indices])*self.dLam,dim=0).detach().numpy()
        U_star_B = np.zeros(n)
        E_B = np.random.exponential(scale=1.0, size=n)
        U_index_B = (Lam_Z_B < E_B).sum(axis=0)
        U_star_B[U_index_B >= n] = np.inf
        U_star_B[U_index_B<n] = self.sorted_times[U_index_B[U_index_B<n]]
        n2 = torch.sum((self.epsilon == 2)[:,0]).item()
        n2_B = (U_index_B>= n).sum()
        U2_B_index = np.random.randint(0,n2,size=n2_B)
        U2_dic = torch.where((self.epsilon == 2)[:,0])[0]
        U_star_B[U_index_B >= n] = self.T[U2_dic[U2_B_index]][:,0]
        u_C_B = torch.rand(n)
        C_B = self.sorted_times[torch.minimum(((1-self.Ghat[self.sort_indices])<u_C_B).sum(axis=0),torch.tensor(n-1))]
        T_B = torch.tensor(np.minimum(U_star_B, C_B)).float().reshape(-1, 1)
        Delta_B = torch.tensor(U_star_B <= C_B).float().reshape(-1, 1)
        epsilon_B = (torch.tensor(2-(U_index_B<n)).reshape(-1, 1)*(Delta_B)).float()
        Z_B = torch.zeros(data.Z.shape)
        cov_B = torch.zeros(data.cov.shape)
        cov_B[U_index_B<n] = data.cov[index_B][U_index_B<n]
        cov_B[U_index_B>=n] = data.cov[U2_dic[U2_B_index]]
        t_vec = T_B
        if len(data.Z.shape) == 3:
            time_breaks_tensor = torch.as_tensor(self.time_breaks, dtype=t_vec.dtype, device=t_vec.device)
            interval_indices = torch.bucketize(t_vec.squeeze(-1), time_breaks_tensor, right=True) - 1
            Z_B = cov_B[:,interval_indices,:].permute(1, 0, 2)
        if len(data.Z.shape) == 2:
            Z_B = cov_B
        T_B = untie_tensor_with_preprocessing(T_B)
        if len(data.Z.shape) == 2:
            n = data.Z.shape[0]
            d = data.Z.shape[1]
            TX_B = torch.zeros((n, n, d+1))
            TX_B[:,:,0] = T_B
            for i in range(d):
                TX_B[:, :, i+1] = Z_B[:, i].reshape(1, n)
        if len(data.Z.shape) == 3:
                TX_B = torch.zeros((data.Z.shape[0], data.Z.shape[1], data.Z.shape[2]+1))
                TX_B[:,:,0] = T_B
                TX_B[:, :, 1:] = Z_B
        data_B = cr_data(T_B, Delta_B, epsilon_B, Z_B, TX_B,cov_B,self.time_breaks)
        self.to(device1)
        return data_B
    def significance_test(self,cov_index,B_seeds,learning_rate = 1e-3, num_epoch = 300, num_epoch_ = num_B, num_epoch_B = num_B, learning_rate_B = learning_B, Nets = [create_net_class(6, 12, 3)],max_batch_size = 2000 ,valid_rate = 0.5,Nets_null = [create_net_class(5, 12, 3)],log = None,num_model = 5, num_model_B = 5):
        # Perform significance test for a specific covariate.
        # cov_index: index of the covariate to be tested.
        # B_seeds: list of bootstrap seeds.
        # learning_rate: learning rate for initial model training.
        # num_epoch: number of epochs for initial model training.
        # num_epoch_: number of epochs for model training during test statistic calculation.
        # num_epoch_B: number of epochs for model training during bootstrap test statistic calculation.
        # learning_rate_B: learning rate for model training during test statistic calculation.
        # Nets: list of neural network classes for the main model.
        # max_batch_size: maximum batch size for training.
        # valid_rate: proportion of data used for validation.
        # Nets_null: list of neural network classes for the null model.
        # log: file path for logging results.
        # num_model: number of random initializations for model training during test statistic calculation.
        # num_model_B: number of random initializations for model training during bootstrap test statistic calculation
        start_time = time.time()
        n = self.n
        batch_num = math.ceil(n/max_batch_size)
        if self.data1 == None:
            self.data1 = self.remove_column(cov_index)
        if self.data1.models['nn_tv'] == None:
            self.data1.nn_tv(Nets = Nets_null,learning_rate1=learning_rate,batch_num=batch_num,best_model_seed_output =True,num_epoch=num_epoch)
        num_valid = int(valid_rate*n)
        batches,data_valid = train_valid(self, num_valid, batch_num)
        batches1, data_valid1 = train_valid(self.data1, num_valid, batch_num)
        loss1_list = []
        loss2_list = []
        for model_seed in range(num_model):
            loss_valid = train_DP(model_seed, Nets_null[0], MyLoss, num_epoch_, learning_rate_B, batches1, data_valid1, time_varying = True)[2]
            loss1_list.append(loss_valid)
            loss_valid = train_DP(model_seed, Nets[0], MyLoss, num_epoch_, learning_rate_B, batches, data_valid, time_varying = True)[2]
            loss2_list.append(loss_valid)
        l1 = (np.array(loss1_list)[~np.isnan(np.array(loss1_list))]).min()
        l2 = (np.array(loss2_list)[~np.isnan(np.array(loss2_list))]).min()
        plr_si = 2*(l1-l2)
        end_time = time.time()
        self.plr_si = ((plr_si,end_time-start_time))
        if log is not None:
            with open(log, 'a') as f:
                output = f"Significance test statistics = {plr_si}.\n"
                output+= f"Time taken for statistics = {end_time-start_time}.\n"
                f.write(output + '\n'+ '-'*50 + '\n')
        for B_seed in B_seeds:
            start_time = time.time()
            data_B = self.data1.Significance_data_B(self,B_seed,Nets_null,learning_rate)
            data_B1 = data_B.remove_column(cov_index)
            batches_B,data_valid_B = train_valid(data_B, num_valid, batch_num)
            batches1_B, data_valid1_B = train_valid(data_B1, num_valid, batch_num)
            loss1_list = []
            loss2_list = []
            for model_seed in range(num_model_B):
                loss_valid = train_DP(model_seed, Nets_null[0], MyLoss, num_epoch_B, learning_rate_B, batches1_B, data_valid1_B, time_varying = True)[2]
                loss1_list.append(loss_valid)
                loss_valid = train_DP(model_seed, Nets[0], MyLoss, num_epoch_B, learning_rate_B, batches_B, data_valid_B, time_varying = True)[2]
                loss2_list.append(loss_valid)
            l1 = (np.array(loss1_list)[~np.isnan(np.array(loss1_list))]).min()
            l2 = (np.array(loss2_list)[~np.isnan(np.array(loss2_list))]).min()
            plr_si_B = 2*(l1-l2)
            end_time = time.time()
            self.plr_si_B.append((plr_si_B,end_time-start_time))
            if log is not None:
                with open(log, 'a') as f:
                    output = f"Bootstrap seed is {B_seed}.\n"
                    output += f"Significance test statistics = {plr_si_B}.\n"
                    output+= f"Time taken for Bootstrap statistics = {end_time-start_time}.\n"
                    f.write(output + '\n')
def patial_likelihood(data, g_value):
    # Calculate the partial likelihood for the given data and g_value.
    n = data.n
    R = data.R
    epsilon = data.epsilon
    Delta = data.Delta
    l = ((g_value.diag()-torch.log(torch.sum(R*torch.exp(g_value),axis=1)))*((2-epsilon).reshape(n))*(Delta.reshape(n))).sum()
    return l            

class MyLoss(nn.Module):
    # Custom loss function for the neural network model.
    def __init__(self):
        super().__init__()
        
    def forward(self, data, g_value):
        return -patial_likelihood(data, g_value)

def df2tensor(df_):
    # Convert a DataFrame to a cr_data instance.
    n = len(df_)
    T = torch.tensor(df_['Survival months'].values).reshape((n,1)).float()
    Delta = torch.tensor((df_['COD to site recode']>0).values).reshape((n,1)).float()
    epsilon = torch.tensor((df_['COD to site recode']).values).reshape((n,1)).float()
    Z = torch.tensor(np.array(df_[df_.columns[3:]]).astype(float)).float()
    data = cr_data(T,Delta,epsilon,Z)
    return data

def make_train_step(model, loss_DNN, optimizer,time_varying):
    # Create a training step function for the model.
    # model: the neural network model
    # loss_DNN: the loss function
    # optimizer: the optimizer for training
    # time_varying: boolean indicating if the model is time-varying or not
    if time_varying == False:
        # If the model is not time-varying, use the following training step function.
        def train_step(data_train,data_valid):
            n_train = data_train.n
            n_valid = data_valid.n
            model.train()
            g_train = torch.zeros((n_train,n_train)).to(device) + model(data_train.Z).T
            data_train.check_R()
            loss_train = loss_DNN(data_train, g_train)
            g_valid = torch.zeros((n_valid,n_valid)).to(device) + model(data_valid.Z).T
            data_valid.check_R()
            loss_valid = loss_DNN(data_valid, g_valid).item()
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss_valid
    else:
        # If the model is time-varying, use the following training step function.
        def train_step(data_train,data_valid):
            model.train()
            data_train.check_TX()
            g_train = model(data_train.TX)[:,:,0]
            data_train.check_R()
            loss_train = loss_DNN(data_train, g_train)
            data_valid.check_TX()
            g_valid = model(data_valid.TX)[:,:,0]
            data_valid.check_R()
            loss_valid = loss_DNN(data_valid, g_valid).item()
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss_valid
    return train_step

def split_tensor_evenly(data_train, batch_nums):
    # Split the data into batches of approximately equal size.
    n = data_train.n  
    batch_size = n // batch_nums  
    remainder = n % batch_nums  
    batches = []
    start_idx = 0
    for i in range(remainder):
        end_idx = start_idx + batch_size + 1
        batches.append(data_train[start_idx:end_idx])
        start_idx = end_idx
    for i in range(remainder, batch_nums):
        end_idx = start_idx + batch_size
        batches.append(data_train[start_idx:end_idx])
        start_idx = end_idx
    return batches

def train_valid(data, num_valid, batch_num):
    # Split the data into training and validation sets.
    torch.manual_seed(99999)
    sample_index = torch.randperm(data.n)
    valid_index = sample_index[:num_valid]
    train_index = sample_index[num_valid:]
    data_train = data[train_index]
    data_valid = data[valid_index]
    batches = split_tensor_evenly(data_train, batch_num)
    for batch in batches:
        batch.data_process()
    data_valid.data_process()
    return batches,data_valid

def train_DP(model_seed, Net, My_loss, num_epoch, learning_rate, batches, data_valid, time_varying, print_loss = False, best_epoch_index = False):
    # Train the model using the specified parameters.
    # model_seed: random seed for model initialization
    # Net: the neural network model class
    # My_loss: the loss function
    # num_epoch: maximum number of epochs for training
    # learning_rate: learning rate for the optimizer
    # batches: list of training batches
    # data_valid: validation data
    torch.manual_seed(model_seed)
    model = Net().to(device)
    loss_DNN = My_loss()
    optimizer = torch.optim.Adam([{'params':model.module.parameters(),'lr':learning_rate}])
    train_step = make_train_step(model, loss_DNN, optimizer, time_varying)
    loss_valid_list = []
    best_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(num_epoch):
        time1 = time.time()
        for batch in batches:
            loss_valid = train_step(batch,data_valid)
        if np.isnan(loss_valid):
            break
        loss_valid_list.append(loss_valid)
        
        # Save best model state
        if loss_valid < best_loss:
            best_loss = loss_valid
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
        
        time2 = time.time()
        if print_loss:
            print(f"Epoch {epoch+1}/{num_epoch}, Loss: {loss_valid:.4f}, Time: {time2-time1:.2f}s")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        final_loss = best_loss
    else:
        final_loss = loss_valid if not np.isnan(loss_valid) else float('inf')
    if best_epoch_index: 
        return loss_valid_list, model, final_loss, best_epoch
    else:
        return loss_valid_list, model, final_loss

def S2hat(data, g_value):
    # Calculate the estimates of survival function at the time points.
    data.G_hat()
    data.Riskmatrix()
    S = torch.mean(data.R*torch.exp(g_value),axis=1)
    return S

def Lam0hat(data, g_value_t):
    # Calculate the estimates of the baseline hazard function at the time points.
    n = data.n
    S2hat0 = S2hat(data, g_value_t-g_value_t.mean(axis = 1).reshape((n,1))).reshape((n,1))
    Lam_hat = ((data.T<=(data.T).T)*(2-data.epsilon)*data.Delta/S2hat0.cpu()).mean(axis = 0)
    return Lam_hat

def estimation(data, batch_num ,Nets, My_loss = MyLoss, random_num = 10, num_epoch = 300, learning_rate1 = 0.001, num_valid_max_estimation = 1000, time_varying = True, best_model_seed_output = False):
    # Estimate the model parameters using the specified parameters.
    # data: the input data
    # batch_num: number of batches for training
    # Nets: list of neural network classes
    # My_loss: the loss function
    # random_num: number of random seeds for model initialization
    # : maximum number of epochs for training
    # learning_rate: learning rate for the optimizer
    # num_valid_max_estimation: maximum number of validation samples
    # time_varying: boolean indicating if the model is time-varying or not
    num_valid = np.minimum(num_valid_max_estimation,int(0.25*data.n))
    batches,data_valid = train_valid(data, num_valid, batch_num)
    loss_list = []
    model_list = []
    epoch_list = []
    if best_model_seed_output:
            for Net_i, Net in enumerate(Nets):
                for model_seed in range(random_num):
                    if len(Nets)>1:
                        learning_rate = learning_rate1[Net_i]
                    else:
                        learning_rate = learning_rate1
                    loss_valid_list, model, loss_valid, best_epoch = train_DP(model_seed, Net, My_loss, num_epoch, learning_rate, batches, data_valid, time_varying,best_epoch_index = best_model_seed_output)
                    loss_list.append(loss_valid)
                    model_list.append(model)
                    epoch_list.append(best_epoch)
    else:
        for Net_i, Net in enumerate(Nets):
            for model_seed in range(random_num):
                if len(Nets)>1:
                    learning_rate = learning_rate1[Net_i]
                else:
                    learning_rate = learning_rate1
                loss_valid_list, model, loss_valid = train_DP(model_seed, Net, My_loss, num_epoch, learning_rate, batches, data_valid, time_varying)
                loss_list.append(loss_valid)
                model_list.append(model)
    loss_array = np.array(loss_list)
    loss_array = loss_array[~np.isnan(loss_array)]
    best_index = loss_array.argmin()
    model = model_list[best_index]
    if best_model_seed_output:
        best_best_epoch = epoch_list[best_index]
        return model,best_index,best_best_epoch
    else:
        return model
def cubic_spline(q,u,m):
    # Create a cubic spline basis matrix.
    # q: number of knots, m: maximum value of T, u: input
    # q+4: number of base of spline, u: input
    T = np.around(np.linspace(0,m.cpu(),q),2)
    u = u.cpu().numpy()
    n = len(u)
    B = np.zeros((n,q+4))
    for i in range(n):
        B[i,0] = 1
        B[i,1] = u[i]
        B[i,2] = u[i]**2
        B[i,3] = u[i]**3
        for jj in range(q):
            j = jj+4
            if u[i] > T[jj]:
                B[i,j] = (u[i] - T[jj])**3
            else:
                B[i,j] = 0
    return torch.tensor(B).to(device)

