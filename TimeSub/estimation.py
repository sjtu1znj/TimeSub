import torch
import numpy as np
from lifelines import KaplanMeierFitter  
import torch.nn as nn
import random
from torch.autograd import Variable
device="cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    # This function sets the random seed for various libraries to ensure reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(0)

class cr_data:
    # competing risk data.
    def __init__(self,T,Delta,epsilon,Z):
        # T: survival time, Delta: event indicator, epsilon: cause of event, Z: covariates
        self.T = T
        self.Delta = Delta
        self.epsilon = epsilon
        self.Z = Z
        self.n = len(T)
        self.Ghat = None
        self.TX = None
        self.R = None
        self.cov_dim = Z.shape[1]
    def __getitem__(self, index):
        # return a new cr_data instance containing the data at index index
        return cr_data(
            T=self.T[index],
            Delta=self.Delta[index],
            epsilon=self.epsilon[index],
            Z=self.Z[index]
        )
    def G_hat(self):
        # G_hat is the estimated cumulative incidence function for the censored event.
        T = self.T
        delta = self.Delta
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=1-delta)
        km = kmf.survival_function_["KM_estimate"]
        Ghat = torch.tensor(km.loc[T[:,0]].values.reshape(len(T[:,0]),1))
        self.Ghat = Ghat

    
    def Riskmatrix(self):
        # Risk matrix is a matrix that indicates the risk set at each time point.
        if self.Ghat == None:
            self.G_hat()
        T = self.T.to(device)
        Ghat = self.Ghat.to(device)
        Delta = self.Delta.to(device)
        epsilon = self.epsilon.to(device)
        n = len(T)
        G = Ghat/Ghat.T
        G[torch.isnan(G)] = 0
        G[torch.isinf(G)] =-0
        R1 = T <= T.T
        R2 = torch.minimum(T >= T.T,torch.ones((n,1)).to(device)@(epsilon-1).T)
        R2 = torch.minimum(R2,torch.ones((n,1)).to(device)@(Delta).T)*G
        R = torch.maximum(R1,R2)
        self.R = R

    
    def X2TX(self):
        # TX is a tensor that combines the survival time and covariates.
        d = self.cov_dim
        n = self.n
        T_test = self.T.to(device)
        X_test = self.Z.to(device)
        TX_test = torch.zeros((n, n, d+1)).to(device)
        TX_test[:, :, 0] = T_test
        for i in range(d):
            TX_test[:,:,1+i] = X_test[:,i].reshape((1,n))
        self.TX = TX_test
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
        newZ = torch.cat((self.Z[:,:column_index],self.Z[:,column_index+1:]),axis=1)
        return cr_data(
            T=self.T,
            Delta=self.Delta,
            epsilon=self.epsilon,
            Z=newZ
        )
        self.data_process()

class cr_data_no_censor(cr_data):
    # competing risk data without censoring.
    def __init__(self, T, epsilon, Z):
        super().__init__(T, None, epsilon, Z)
        self.Delta = None  # Delta attribute is set to None

    def __getitem__(self, index):
        # 返回一个新的 cr_data_no_Delta 实例，包含索引 index 处的数据
        return cr_data_no_censor(
            T=self.T[index],
            epsilon=self.epsilon[index],
            Z=self.Z[index]
        )

    def G_hat(self):
        T = self.T
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=torch.ones(len(T)))  # Assuming all events are observed
        km = kmf.survival_function_["KM_estimate"]
        Ghat = torch.tensor(km.loc[T[:, 0]].values.reshape(len(T[:, 0]), 1))
        self.Ghat = Ghat

    def Riskmatrix(self):
        T = self.T.to(device)
        epsilon = self.epsilon.to(device)
        n = len(T)
        R1 = T <= T.T
        R2 = torch.minimum(T >= T.T, torch.ones((n, 1)).to(device) @ (epsilon - 1).T)
        R = torch.maximum(R1, R2)
        self.R = R

    def to(self, device):
        self.T = self.T.to(device)
        self.epsilon = self.epsilon.to(device)
        self.Z = self.Z.to(device)
        if self.Ghat != None:
            self.Ghat = self.Ghat.to(device)
        if self.TX != None:
            self.TX = self.TX.to(device)
        if self.R != None:
            self.R = self.R.to(device)

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

def train_DP(model_seed, Net, My_loss, num_epochs, learning_rate, batches, data_valid, time_varying, print_loss = False):
    # Train the model using the specified parameters.
    # model_seed: random seed for model initialization
    # Net: the neural network model class
    # My_loss: the loss function
    # num_epochs: maximum number of epochs for training
    # learning_rate: learning rate for the optimizer
    # batches: list of training batches
    # data_valid: validation data
    torch.manual_seed(model_seed)
    model = Net().to(device)
    loss_DNN = My_loss()
    optimizer = torch.optim.Adam([{'params':model.module.parameters(),'lr':learning_rate}])
    train_step = make_train_step(model, loss_DNN, optimizer, time_varying)
    loss_valid_list = []
    for epoch in range(num_epochs):
        for batch in batches:
            loss_valid = train_step(batch,data_valid)
        if np.isnan(loss_valid):
            break
        loss_valid_list.append(loss_valid)
        if print_loss:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_valid:.4f}")
    return loss_valid_list, model, loss_valid

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

def estimation(data, batch_num ,Nets, My_loss = MyLoss, random_num = 10, num_epochs = 300, learning_rate = 0.001, num_valid_max_estimation = 1000, time_varying = True):
    # Estimate the model parameters using the specified parameters.
    # data: the input data
    # batch_num: number of batches for training
    # Nets: list of neural network classes
    # My_loss: the loss function
    # random_num: number of random seeds for model initialization
    # num_epochs: maximum number of epochs for training
    # learning_rate: learning rate for the optimizer
    # num_valid_max_estimation: maximum number of validation samples
    # time_varying: boolean indicating if the model is time-varying or not
    num_valid = np.minimum(num_valid_max_estimation,int(0.25*data.n))
    batches,data_valid = train_valid(data, num_valid, batch_num)
    loss_list = []
    model_list = []
    for Net in Nets:
        for model_seed in range(random_num):
            loss_valid_list = train_DP(model_seed, Net, My_loss, num_epochs, learning_rate, batches, data_valid, time_varying)[0]
            best_num_epochs = np.argmin(np.array(loss_valid_list))+1
            model, loss_valid = train_DP(model_seed, Net, My_loss, best_num_epochs, learning_rate, batches, data_valid, time_varying)[1:]
            loss_list.append(loss_valid)
            model_list.append(model)
    loss_array = np.array(loss_list)
    loss_array = loss_array[~np.isnan(loss_array)]
    best_index = loss_array.argmin()
    model = model_list[best_index]
    return model

def cubic_spline(q,u,m):
    # Create a cubic spline basis matrix.
    # q: number of knots, m: maximum value of T, u: input
    # q+4: number of base of spline, u: input
    T = np.around(np.linspace(0,m,q),2)
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

def patial_likelihood_spline(data, B,Gamma):
    # Calculate the partial likelihood for the given data and spline basis.
    if data.R == None:
        data.Riskmatrix()
    data.to(device)
    epsilon = data.epsilon
    n = data.n
    Delta = data.Delta
    R = data.R
    g_value = (Gamma@B.T).T@data.Z.T
    l = ((g_value.diag()-torch.log(torch.sum(R*torch.exp(g_value),axis=1)))*((2-epsilon).reshape(n))*(Delta.reshape(n))).sum()
    return l

def estimation_spline(data, num_spline):
    # Estimate the spline coefficients using the given data and number of splines.
    d = data.Z.shape[1]
    m = data.T.max()
    B = cubic_spline(num_spline,data.T[:,0].cpu(),m.cpu()).float()
    Gamma = Variable(torch.zeros((d,num_spline+4)).to(device), requires_grad=True)
    optimizer = torch.optim.LBFGS([Gamma])
    num_epochs = 1000
    def closure():
        optimizer.zero_grad()
        loss = -patial_likelihood_spline(data, B,Gamma)
        loss.backward()
        return loss
    for epoch in range(num_epochs):
        loss_now = -patial_likelihood_spline(data, B,Gamma)
        if epoch>1: 
            if loss_now-loss_last>=-0.01:
                break
        optimizer.step(closure)
        loss_last = -patial_likelihood_spline(data, B,Gamma)
        if np.isnan(loss_last.item()):
            break
    return Gamma

def n_fold_cv(seed,fold,n):
    # Perform n-fold cross-validation.
    torch.manual_seed(seed)
    index_list = []
    index = torch.randperm(n)
    fold_n = int(n/fold)
    for i in range(fold-1):
        valid_index = index[i*fold_n:(i+1)*fold_n]
        train_index = torch.cat([index[:i*fold_n],index[(i+1)*fold_n:]])
        index_list.append([train_index,valid_index])
    index_list.append((index[:fold_n*(fold-1)],index[fold_n*(fold-1):]))
    return index_list

def spline_cv(data, num_spline_list = [2,5,10,20],seed = 0, fold= 5):
    # Perform cross-validation for spline estimation.
    n = data.n
    loss_num_list = []
    Gamma_num_list = []
    for num_spline in num_spline_list:
        loss_list = []
        Gamma_list = []
        for train_index,valid_index in n_fold_cv(seed,fold,n):
            m = data.T.max()
            Gamma = estimation_spline(data[train_index], num_spline)
            data_valid = data[valid_index]
            data_valid.G_hat()
            data_valid.to(device)
            B_test = cubic_spline(num_spline,data_valid.T[:,0].cpu(),m.cpu()).float()
            loss_test = -patial_likelihood_spline(data_valid, B_test,Gamma).item()
            loss_list.append(loss_test)
            Gamma_list.append(Gamma.detach())
        loss_num_list.append(np.array(loss_list).mean())
        Gamma_num_list.append(torch.stack(Gamma_list).mean(axis = 0))
    loss_array = np.array(loss_num_list)
    loss_array = loss_array[~np.isnan(loss_array)]
    best_index = loss_array.argmin()
    best_Gamma = Gamma_num_list[best_index]
    return best_Gamma