import torch
from lifelines import KaplanMeierFitter  
import numpy as np
device="cuda" if torch.cuda.is_available() else "cpu"

def TPR(data, g_torch_test):
    # Calculate True Positive Rate (TPR) using the risk matrix
    data.Riskmatrix()
    data.to("cpu")
    n = data.n
    M1 = g_torch_test.reshape((n,1,n))
    M2 = g_torch_test.reshape((n,n,1))
    R = data.R
    Numerator = ((M1>M2)*R*torch.exp(M1)).sum(axis = 2)
    Denominator = (R*torch.exp(M1)).sum(axis = 2)
    TP = Numerator/Denominator
    data.to(device)

    return TP.to(device)

def FPR(data, g_torch_test):
    # Calculate False Positive Rate (FPR) using the risk matrix
    data.Riskmatrix()
    data.to("cpu")
    n = data.n
    M1 = g_torch_test.reshape((n,1,n))
    M2 = g_torch_test.reshape((n,n,1))
    R = data.R.reshape((n,1,n))
    Numerator = ((M1>M2)*R).sum(axis = 2)
    Denominator = (R).sum(axis = 2)
    FP = Numerator/Denominator
    data.to(device)
    return FP.to(device)

def prediction_ability(data,g_torch_test):
    # Calculate the prediction ability (the C-index and AUC)
    S_hat = KaplanMeierFitter()
    epsilon_test = data.epsilon
    Delta_test = data.Delta
    T_test = data.T
    eff_index = ((epsilon_test==1)&(Delta_test==1)).cpu().numpy()
    in_rate = (eff_index.sum()/Delta_test.sum()).cpu().numpy()
    T_ = T_test[eff_index].cpu().numpy()
    Delta_ = Delta_test[eff_index].cpu().numpy()
    S_hat.fit(T_,event_observed=Delta_)
    t_ = np.array(S_hat.survival_function_.index)
    n1 = len(T_)
    n2 = len(t_)
    if n1>n2+1:
        a = T_.reshape((n1,1))
        drop_index = np.where((a==a.T).sum(axis= 0)>1)[0]
        T_ = np.delete(T_,drop_index)
        Delta_ = np.delete(Delta_,drop_index)
        S_hat.fit(T_,event_observed=Delta_)
        t_ = np.array(S_hat.survival_function_.index)
    else:
        drop_index = []
    S_finite = np.array(S_hat.survival_function_['KM_estimate'])
    S_ = in_rate*S_finite+(1-in_rate)
    f_ = ((S_[:-1]-S_[1:])/(t_[1:]-t_[:-1]))
    w_ = torch.tensor(2*S_[:-1]*f_/(1-(S_[-2])**2)).to(device)
    delta_t_ = torch.tensor(t_[1:]-t_[:-1]).to(device)
    TP = TPR(data, g_torch_test).sort(axis = 1).values
    FP = FPR(data, g_torch_test).sort(axis = 1).values
    n  = data.n
    auc_array = torch.zeros(n).to(device)
    for j in range(n):
        auc_array[j]=torch.trapz(y=TP[j], x=FP[j])
    C_tau = (torch.tensor(np.delete(auc_array.cpu().numpy()[eff_index[:,0]],(drop_index))).to(device)[torch.tensor(T_).sort().indices]*w_*delta_t_)[:-1].sum()
    y = auc_array[T_test[:,0].sort().indices][:-1]
    return C_tau.item(), y