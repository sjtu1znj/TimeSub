import sys
sys.path.append("..")
from TimeSub.estimation import train_DP, train_valid, Lam0hat, MyLoss, set_seed, cr_data_no_censor
import numpy as np 
import itertools
import torch
import torch.nn as nn
from scipy.interpolate import UnivariateSpline
import time
import random
import math
device="cuda" if torch.cuda.is_available() else "cpu"

# This function estimates the nonparametric function under the null hypothesis of structure test.
# This function also estimates the baseline hazard function and its inverse.
# All estimator mentioned above are used to generate the new data (bootstrap data) under the null hypothesis.
def estimation_null_structure(data, batch_num ,Nets, My_loss = MyLoss, random_num = 5, num_epochs = 400, learning_rate = 0.001, num_valid_max_estimation = 1000):
    # data: the data object containing the training data and the validation data
    # batch_num: the number of batches to use for training
    # Nets: a list of neural network architectures to use for training
    # My_loss: the loss function to use for training
    # random_num: the number of random seeds to use for training
    # num_epochs: the number of epochs to train for
    # learning_rate: the learning rate to use for training
    # num_valid_max_estimation: the maximum number of validation samples to use for training
    num_valid = np.minimum(num_valid_max_estimation,int(0.25*data.n))
    batches,data_valid = train_valid(data, num_valid, batch_num)
    loss_list = []
    model_list = []
    for Net in Nets:
        for model_seed in range(random_num):
            loss_valid_list = train_DP(model_seed, Net, My_loss, num_epochs, learning_rate, batches, data_valid, False)[0]
            best_num_epochs = np.argmin(np.array(loss_valid_list))+1
            model, loss_valid = train_DP(model_seed, Net, My_loss, best_num_epochs, learning_rate, batches, data_valid, False)[1:]
            loss_list.append(loss_valid)
            model_list.append(model)
    loss_array = np.array(loss_list)
    loss_array = loss_array[~np.isnan(loss_array)]
    best_index = loss_array.argmin()
    model = model_list[best_index]
    g_value = (torch.zeros((data.n,data.n)).to(device)+model(data.Z.to(device))).T
    Lam_hat = Lam0hat(data, g_value)
    x = data.T.sort(axis = 0).values[:,0].detach().numpy()
    y2 = Lam_hat[data.T.sort(axis = 0).indices[:,0]].detach().numpy()
    Lam0_h = UnivariateSpline(x, y2, s=0.2)
    x_in = np.linspace(0,x.max(),1000)
    y_in = np.sort(Lam0_h(x_in))
    Lam0_in = UnivariateSpline(y_in, x_in, s=0.2)
    return model,Lam0_h,Lam0_in

# This function calculates the structure test statistics for the given data.
def estimation_structure(data, batch_num ,Nets1, Nets2, My_loss = MyLoss, random_num = 5, num_epochs = 400, learning_rate = 0.001, num_valid_max_test = 3000):
    # data: the data object containing the training data and the validation data
    # batch_num: the number of batches to use for training
    # Nets1: a list of neural network architectures to use for estimating pesudo partial likelihood under the null hypothesis
    # Nets2: a list of neural network architectures to use for estimating pesudo partial likelihood in full function space
    # My_loss: the loss function to use for training
    # random_num: the number of random seeds to use for training
    # num_epochs: the number of epochs to train for
    # learning_rate: the learning rate to use for training
    # num_valid_max_test: the maximum number of validation samples to use for training
    num_valid = np.minimum(num_valid_max_test,int(0.5*data.n))
    batches,data_valid = train_valid(data, num_valid, batch_num)
    loss1_list = []
    loss2_list = []
    for Net in Nets1:
        for model_seed in range(random_num):
            loss_valid_list = train_DP(model_seed, Net, My_loss, num_epochs, learning_rate, batches, data_valid,False)[0]
            loss1_list.append(loss_valid_list)
    for Net in Nets2:
        for model_seed in range(random_num):
            loss_valid_list = train_DP(model_seed, Net, My_loss, num_epochs, learning_rate, batches, data_valid,True)[0]
            loss2_list.append(loss_valid_list)
    
    l1 = np.array(list(itertools.chain(*loss1_list))).min()
    l2 = np.array(list(itertools.chain(*loss2_list))).min()
    lambda_t = 2*(l1-l2)
    return lambda_t

def patial_likelihood_B(data, g_value):
    # This function calculates the partial likelihood for the bootstrap data.
    n = data.n
    R = data.R
    epsilon = data.epsilon
    l_B = ((g_value.diag()-torch.log(torch.sum(R*torch.exp(g_value),axis=1)))*((2-epsilon).reshape(n))).sum()
    return l_B

class Loss_B(nn.Module):
    # This class defines the loss function for the bootstrap data.
    def __init__(self):
        super().__init__()
        
    def forward(self, data, g_value):
        return -patial_likelihood_B(data, g_value)

# This function calculates the p-value for the structure test.
def structure_test(data, batch_num, Nets1, Nets2, 
                   Loss_estimation = MyLoss, Loss_test = Loss_B, 
                   random_num_estimation = 5, random_num_test = 5,
                   num_epochs_estimation = 400, num_epochs_test = 400,
                   learning_rate_estimation = 0.001, learning_rate_test = 0.001,
                   num_valid_max_estimation = 1000, num_valid_max_test = 3000,
                   print_info = False):
    # data: the data object containing the training data and the validation data
    # batch_num: the number of batches to use for training
    # Nets1: a list of neural network architectures to use for estimating pesudo partial likelihood under the null hypothesis
    # Nets2: a list of neural network architectures to use for estimating pesudo partial likelihood in full function space
    # Loss_estimation: the loss function to use for calculate pesudo partial likelihood with original data
    # Loss_test: the loss function to use for calculate pesudo partial likelihood with bootstrap data
    # random_num_estimation: the number of random seeds to use for estimation
    # random_num_test: the number of random seeds to use for test
    # num_epochs_estimation: the number of epochs to train for estimation
    # num_epochs_test: the number of epochs to train for test
    # learning_rate_estimation: the learning rate to use for estimation
    # learning_rate_test: the learning rate to use for test
    # num_valid_max_estimation: the maximum number of validation samples to use for estimation
    # num_valid_max_test: the maximum number of validation samples to use for test
    # print_info: whether to print the training time and p-value
    T1 = time.time()
    model_B,Lam0_h,Lam0_in = estimation_null_structure(data, batch_num ,Nets1, Loss_estimation, random_num_estimation, num_epochs_estimation, learning_rate_estimation, num_valid_max_estimation)
    T2 = time.time()
    if print_info:
        print("model_B training time:",T2-T1,"seconds")
    threshold_ep = data.T.sort(axis = 0).values[:,0].cpu().detach().numpy().max()
    lambda_t = estimation_structure(data, batch_num ,Nets1, Nets2, Loss_estimation, random_num_test, num_epochs_test, learning_rate_test, num_valid_max_test)
    T3 = time.time()
    if print_info:
        print('lambda t =',lambda_t)
        print("model_B training time:",T3-T2,"seconds")
    lam_B_list = []
    n = data.n
    for boot_seed in range(10100,10200):
        T1 = time.time()
        set_seed(boot_seed)
        index_B = np.random.randint(0,n,size=n)
        X_B = data.Z[index_B].to(device)
        u_B = torch.rand(n).to(device)
        hazard_B = torch.exp((model_B(X_B)[:,0]-model_B(data.Z.to(device)).mean()).detach())
        epsilon_B = ((-(torch.log(1-u_B)/hazard_B).cpu()>Lam0_h(threshold_ep).item())+1).to(device).reshape((n,1)).float()
        U1_B  = torch.tensor(Lam0_in(-(torch.log(1-u_B)/hazard_B).cpu())).to(device).reshape((n,1)).float()
        U1_B[epsilon_B==2]=1
        U_B = (U1_B-U1_B.min())
        data_B = cr_data_no_censor(U_B, epsilon_B, X_B)
        data_B.to("cpu")
        lambda_B = estimation_structure(data_B, batch_num ,Nets1, Nets2, Loss_test, random_num_test, num_epochs_test, learning_rate_test, num_valid_max_test)
        lam_B_list.append(lambda_B)
        T2 = time.time()
        if print_info:
            print('lambda B :',lambda_B)
            print("model_B training time:",T2-T1,"seconds")
    if len(np.where(lambda_t<np.sort(np.array(lam_B_list)))[0]) == 0:
        p = np.array(0.0)
    else:
        p = (1-np.where(lambda_t<np.sort(np.array(lam_B_list)))[0].min()/100)
    if print_info:
        print('p value:',p)
    return lambda_t, p, lam_B_list

# This function estimates the nonparametric function under the null hypothesis of significance test.
# This function also estimates the baseline hazard function.
# All estimator mentioned above are used to generate the new data (bootstrap data) under the null hypothesis.
def estimation_null_significance(data, batch_num ,Nets, My_loss = MyLoss, random_num = 5, num_epochs = 300, learning_rate = 0.001, num_valid_max_estimation = 1000):
    # data: the data object containing the training data and the validation data
    # batch_num: the number of batches to use for training
    # Nets: a list of neural network architectures to use for training
    # My_loss: the loss function to use for training
    # random_num: the number of random seeds to use for training
    # num_epochs: the number of epochs to train for
    # learning_rate: the learning rate to use for training
    # num_valid_max_estimation: the maximum number of validation samples to use for training
    num_valid = np.minimum(num_valid_max_estimation,int(0.25*data.n))
    batches,data_valid = train_valid(data, num_valid, batch_num)
    loss_list = []
    model_list = []
    for Net in Nets:
        for model_seed in range(1,random_num+1):
            loss_valid_list = train_DP(model_seed, Net, My_loss, num_epochs, learning_rate, batches, data_valid, True)[0]
            if len(loss_valid_list) >0 :
                best_num_epochs = np.argmin(np.array(loss_valid_list))+1
            else:
                best_num_epochs = 1
                print(model_seed,': empty')
            model, loss_valid = train_DP(model_seed, Net, My_loss, best_num_epochs, learning_rate, batches, data_valid, True)[1:]
            loss_list.append(loss_valid)
            model_list.append(model)
    loss_array = np.array(loss_list)
    loss_array = loss_array[~np.isnan(loss_array)]
    best_index = loss_array.argmin()
    model = model_list[best_index]
    data.X2TX()
    g_value = model(data.TX)[:,:,0]
    Lam_hat = Lam0hat(data, g_value)
    x = data.T.sort(axis = 0).values[:,0].detach().numpy()
    y2 = Lam_hat[data.T.sort(axis = 0).indices[:,0]].detach().numpy()
    Lam0_h = UnivariateSpline(x, y2, s=0.2)
    x_new = np.linspace(0,x[-1],1000)
    Y_new = Lam0_h(x_new)
    y_new = (Y_new[1:]-Y_new[:-1])/(x_new[1:]-x_new[:-1])
    lam0_h = UnivariateSpline(x_new[1:], y_new, s=0.2)
    return model,lam0_h

# This function calculates the significance test statistics for the given data.
def estimation_significance(data,data1, batch_num ,Nets1,Nets2,My_loss = MyLoss, random_num = 5, num_epochs = 300, learning_rate = 0.001, num_valid_max_test = 3000):
    # data: the data object containing the training data and the validation data
    # data1: the data object containing the training data and the validation data without the tested covariate
    # batch_num: the number of batches to use for training
    # Nets1: a list of neural network architectures to use for estimating pesudo partial likelihood under the null hypothesis
    # Nets2: a list of neural network architectures to use for estimating pesudo partial likelihood in full function space
    # My_loss: the loss function to use for training
    # random_num: the number of random seeds to use for training
    # num_epochs: the number of epochs to train for
    # learning_rate: the learning rate to use for training
    # num_valid_max_test: the maximum number of validation samples to use for training
    num_valid = np.minimum(num_valid_max_test,int(0.5*data.n))
    batches1,data_valid1 = train_valid(data1, num_valid, batch_num)
    batches,data_valid = train_valid(data, num_valid, batch_num)
    loss1_list = []
    loss2_list = []
    for Net in Nets2:
        for model_seed in range(1,random_num+1):
            loss_valid_list = train_DP(model_seed, Net, My_loss, num_epochs, learning_rate, batches1, data_valid1, True)[0]
            loss1_list.append(np.array(loss_valid_list).min())
    for Net in Nets1:
        for model_seed in range(1,random_num+1):
            loss_valid_list = train_DP(model_seed, Net, My_loss, num_epochs, learning_rate, batches, data_valid, True)[0]
            loss2_list.append(np.array(loss_valid_list).min())
    l1 = np.array(loss1_list).min()
    l2 = np.array(loss2_list).min()
    return 2*(l1-l2)

# This function calculates the p-value for the significance test.
def significance_test(data, cov_index, batch_num, Nets1, Nets2,
                     Loss_estimation = MyLoss, Loss_test = Loss_B, 
                     random_num_estimation = 5, random_num_test = 5,
                     num_epochs_estimation = 300, num_epochs_test = 300,
                     learning_rate_estimation = 0.001, learning_rate_test = 0.001,
                     num_valid_max_estimation = 1000, num_valid_max_test = 3000,
                     print_info = False):
    # data: the data object containing the training data and the validation data
    # cov_index: the index of the covariate to be tested
    # batch_num: the number of batches to use for training
    # Nets1: a list of neural network architectures to use for estimating pesudo partial likelihood under the null hypothesis
    # Nets2: a list of neural network architectures to use for estimating pesudo partial likelihood in full function space
    # Loss_estimation: the loss function to use for calculate pesudo partial likelihood with original data
    # Loss_test: the loss function to use for calculate pesudo partial likelihood with bootstrap data
    # random_num_estimation: the number of random seeds to use for estimation
    # random_num_test: the number of random seeds to use for test
    # num_epochs_estimation: the number of epochs to train for estimation
    # num_epochs_test: the number of epochs to train for test
    # learning_rate_estimation: the learning rate to use for estimation
    # learning_rate_test: the learning rate to use for test
    # num_valid_max_estimation: the maximum number of validation samples to use for estimation
    # num_valid_max_test: the maximum number of validation samples to use for test
    # print_info: whether to print the training time and p-value
    T1 = time.time()
    data1 = data.remove_column(cov_index)
    model_B,lam0_h = estimation_null_significance(data1, batch_num ,Nets2, Loss_estimation, random_num_estimation, num_epochs_estimation, learning_rate_estimation, num_valid_max_estimation)
    T2 = time.time()
    if print_info:
        print("model_B training time:",T2-T1,"seconds")
    lambda_t = estimation_significance(data,data1, batch_num ,Nets1,Nets2, Loss_estimation, random_num_test, num_epochs_test, learning_rate_test, num_valid_max_test)
    T3 = time.time()
    if print_info:
        print('lambda t =',lambda_t)
        print("model_B training time:",T3-T2,"seconds")
    tm = data.T.max().item()
    g_mean_x = data1.T[:,0].sort().values.cpu().detach().numpy()
    g_mean_indices = data1.T[:,0].sort().indices
    data1.X2TX()
    g_mean_y = model_B(data1.TX.to(device)).mean(axis = 1)[:,0][g_mean_indices].cpu().detach().numpy()
    g_mean = UnivariateSpline(g_mean_x, g_mean_y, s=0.2)
    lam_B_list = []
    n = data.n
    for boot_seed in range(10100,10200):
        T1 = time.time()
        set_seed(boot_seed)
        index_B = np.random.randint(0,n,size=n)
        X_B = data.Z[index_B].to(device)
        X_B1 = torch.cat((X_B[:,:cov_index],X_B[:,cov_index+1:]),axis=1)
        U_list = []
        for i_B in range(n):
            z = X_B1[i_B]
            u  = random()
            t_x = torch.linspace(0,tm,1000)
            t_y = torch.tensor(lam0_h(t_x)).to(device)*torch.exp(model_B(torch.cat([t_x.reshape(1,1000).to(device),z.reshape((data1.cov_dim,1)).expand((-1,1000))]).to(device).T)[:,0]-torch.Tensor(g_mean(t_x)).to(device))
            t_x = t_x.to(device)
            integ = torch.tril(torch.ones(1000, 1000)).to(device)@t_y.float()*(t_x[1]-t_x[0])
            if u >= 1-math.exp(-integ[-1]):
                U_i = math.inf
            else:
                t_F = 1-torch.exp(-integ)
                U_i = t_x[torch.where((t_F-u)>0)[0][0].item()]
            U_list.append(U_i)
        U_B = torch.tensor(U_list).float().reshape((n,1))
        epsilon_B = (U_B.isinf()+1.0).reshape((n,1))
        U_B[U_B.isinf()] = 1
        data_B = cr_data_no_censor(U_B, epsilon_B, X_B)
        data_B1 = cr_data_no_censor(U_B, epsilon_B, X_B1)
        data_B.to("cpu")
        lambda_B = estimation_significance(data_B, data_B1, batch_num, Nets1, Nets2, Loss_test, random_num_test, num_epochs_test, learning_rate_test, num_valid_max_test)
        lam_B_list.append(lambda_B)
        T2 = time.time()
        if print_info:
            print('lambda B :',lambda_B)
            print("model_B training time:",T2-T1,"seconds")
    if len(np.where(lambda_t<np.sort(np.array(lam_B_list)))[0]) == 0:
            p = np.array(0.0)   
    else:
        p = (1-np.where(lambda_t<np.sort(np.array(lam_B_list)))[0].min()/100)
    if print_info:
            print('p value:',p)
    return lambda_t, p, lam_B_list