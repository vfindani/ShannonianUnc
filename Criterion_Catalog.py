import csv

import numpy as np
from sklearn.metrics import r2_score
import Entropy
import matplotlib.pyplot as plt
from Entropy import *
# Θέλω όλα τα κριτήρια να έχουν ως μέγιστο το 10 και πάντα να είναι μη αρνητικά


def NSE(observed, simulated):

    observed, simulated = remove_nan_values(observed, simulated)
    error = observed-simulated
    error2 = np.power(error, 2)
    obs_var = observed - np.average(observed)
    obs_var2 = np.power(obs_var, 2)
    nse_value = 1.0 - np.sum(error2)/np.sum(obs_var2)
    nse_value = nse_value + 9.0  # Ο μετασχηματισμός αυτός γίνεται ώστε πάντα το κριτήριο να είναι θετικό
    nse_value = kill_negatives(nse_value)

    return nse_value


def NNSE(observed, simulated):

    observed, simulated = remove_nan_values(observed, simulated)
    error = observed-simulated
    error2 = np.power(error, 2)
    obs_var = observed - np.average(observed)
    obs_var2 = np.power(obs_var, 2)
    nse_value = 1.0 - np.sum(error2)/np.sum(obs_var2)
    nnse = 10.0/(2-nse_value)

    return nnse


def MSE(observed, simulated):

    observed, simulated = remove_nan_values(observed, simulated)
    error = observed-simulated
    error2 = np.power(error, 2)
    mse_value = np.average(error2)
    # Γίνεται ο παρακάτω μετασχηματισμός γιατί θέλουμε όλα τα κριτήρια να έχουν ως βέλτιστη τιμή το 10
    mse_value = 10.0 - mse_value
    mse_value = kill_negatives(mse_value)

    return mse_value


def RMSE(observed, simulated):

    mse_value = MSE(observed, simulated)
    rmse_value = mse_value ** .5

    return rmse_value


def KGE(observed, simulated):

    observed, simulated = remove_nan_values(observed, simulated)
    m_o = np.average(observed)
    m_s = np.average(simulated)
    sigma_o = np.std(observed)
    sigma_s = np.std(simulated)
    alpha = sigma_s/sigma_o
    beta = m_s/m_o
    conva = np.cov(observed, simulated)[0][1]
    r = conva/(sigma_o*sigma_s)
    kge_value = 1.0-np.sqrt((r-1.0)**2+(alpha-1.0)**2+(beta-1.0)**2)
    # https://agrimetsoft.com/calculators/Kling-Gupta%20efficiency
    kge_value = kge_value+9.0
    kge_value = kill_negatives(kge_value)

    return kge_value


def DV(observed, simulated):

    observed, simulated = remove_nan_values(observed, simulated)
    error = simulated-observed
    dv_value = np.sum(error)/np.sum(observed)
    # Γίνεται ο παρακάτω μετασχηματισμός γιατί θέλουμε όλα τα κριτήρια να έχουν ως βέλτιστη τιμή το 10
    dv_value = 10.0*(1 - np.abs(dv_value))
    dv_value = kill_negatives(dv_value)

    return dv_value


def IA(observed, simulated):

    observed, simulated = remove_nan_values(observed, simulated)
    error = observed - simulated
    error2 = np.power(error, 2)
    c = np.abs(simulated-np.average(observed))+np.abs(observed-np.average(observed))
    c2 = np.power(c, 2)
    ia_value = 1.0-np.sum(error2)/np.sum(c2)
    ia_value = ia_value+9.0
    ia_value = kill_negatives(ia_value)

    return ia_value


def R2(observed, simulated):

    # variance2 = (observed-np.average(observed))**2
    # SStot = np.sum(variance2)
    # error2 = (observed - simulated)**2
    # SSres= np.sum(error2)
    # R2 = 1-SSres/SStot
    observed, simulated = remove_nan_values(observed, simulated)
    R2 = r2_score(observed, simulated)
    R2 = R2 + 9.0
    R2 = kill_negatives(R2)

    return R2


def kill_negatives(x):
    y = x
    if x < 0.01:
        y = 0.01
    return y


def just_return_Qsim(observed, simulated):
    return simulated


def just_return_Qobs(observed, simulated):
    return observed


def ExpInf(observed, simulated):
    observed, simulated = remove_nan_values(observed, simulated)
    k = np.size(observed)

    with Entropy.suppress_stdout():
        ExInf = Entropy.Mutual_Information_2D(observed.T, simulated.T, M=int(k/5), L=0, suppress_negatives=True, noise=3, ICA=True)
    return ExInf


def ExpInfLog(observed, simulated):
    observed, simulated = remove_nan_values(observed, simulated)
    observed[observed <= 0] = 0.01
    simulated[simulated <= 0] = 0.01

    # print(simulated)
    observed = np.log(observed)
    simulated = np.log(simulated)

    k = np.size(observed)
    M = max(15, int(k/4))

    ExInf = 0.0
    try:
        with Entropy.suppress_stdout():
            ExInf = Entropy.Mutual_Information_2D(observed.T, simulated.T, M, L=0, suppress_negatives=True, noise=3, ICA=True)
    except Exception:
        print("error: ExpInfLog did not return any value.")
        pass

    return ExInf


def AvInf(X, Q):
    M, L = Entropy.adjustSteps(X, Q)
    Integrated_AvInf = Entropy.Mutual_Information_2D(X, Q, M, L, suppress_negatives=True, noise=3, ICA=False)
    return AvInf


# Η ρουτίνα αυτή αφαιρεί κενές εγγραφές από τον πίνακα observed, για να υπολογισθούν τα κριτήρια και τις αντίστοιχες των
# κενών γραμμές στον πίνακα simulated
def remove_nan_values(observed, simulated):
    observed_0 = observed[~np.isnan(observed)]
    simulated_0 = simulated[~np.isnan(observed)]
    return observed_0, simulated_0
