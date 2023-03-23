import numpy as np
import Hydrological_Models
from timeit import default_timer as timer
from pyDOE2 import *
import matplotlib.pyplot as plt
import Manipulate_Population
from itertools import combinations
import math
import Entropy


# # Η ρουτίνα αυτή υπολογίζει τις αβεβαιότητες σε bits
# # Το C είναι το βήμα έως το οποίο θα μετατοπιστούν οι χρονοσειρές των δεδομένων εισόδου
# # Πχ αν χρονοσειρές εισόδου είναι οι R(t) και PET(t) και επιλέξω C=2 τότε θα δημιουργηθούν οι
# # χρονοσειρές R(t-1), PET(t-1), R(t-2), PET(t-2) και θα εννωθούν όλες μαζί στον πίνακα X1
# def Uncertainties(X, Qobs, Qsim, C):
#
#     Q = np.resize(Qobs, (np.size(Qobs, axis=0), 1))
#
#     # Δημιουργούνται οι υστερημένες χρονοσειρές
#     X1 = X
#     Xlagged = np.zeros((np.size(X, axis=0), np.size(X, axis=1), C))
#     for j in range(0, C):
#         # for i in range(j+1, np.size(X, axis=0)):
#         #     Xlagged[i-j-1, :, j] = X[i, :]
#         for i in range(0, np.size(X, axis=0)-j-1):
#            Xlagged[i+j+1, :, j] = X[i, :]
#         X1 = np.hstack((X1, Xlagged[:, :, j]))
#
#     X2 = np.hstack((X1, Q))
#     phi = EntropyND(X2)-EntropyND(X1)
#     print(EntropyND(X2), EntropyND(X1))
#
#     H_obs = Entropy1D(Qobs)[0]
#     H_sim = Entropy1D(Qsim)[0]
#     JH_obs_sim = Entropy2D(Qobs, Qsim)
#     I_obs_sim = H_obs + H_sim - JH_obs_sim
#     psi = H_obs-I_obs_sim-phi
#
#     return H_obs, (H_obs-phi), I_obs_sim, phi, psi

# Υπολογισμός της σύνθετης (διακριτής) εντροπίας δύο μεταβλητών X1 και X2
# Οι πίνακες Χ1 και Χ2 πρέπει να έχουν μόνο μία διάσταση
# Το αποτέλεσμα της ρουτίνας είναι η βαθμωτή τιμή της σύνθετης εντροπίας
# ο υπολογισμός της εντροπίας γίνεται με bin counting (επισφαλές για συνεχείς μεταβλητές)
def Entropy2D(X1, X2):

    k = 0
    H2D = 0.0

    if np.shape(X1) == np.shape(X2):
        k = np.size(X1, axis=0)

        sigma1 = np.std(X1)
        bin_width1 = 3.73 * sigma1 * k ** (-1 / 4)  # Χρήση της εξίσωσης του Scott
        M1 = int((np.max(X1) - np.min(X1)) / bin_width1)
        sigma2 = np.std(X2)
        bin_width2 = 3.73 * sigma2 * k ** (-1 / 4)  # Χρήση της εξίσωσης του Scott
        M2 = int((np.max(X2) - np.min(X2)) / bin_width2)

        frequency_joint, bins_obs, bins_sim = np.histogram2d(X1, X2, bins=(M1, M2))
        JPMD = frequency_joint / k
        JPMD = JPMD[JPMD != 0]
        H2D = - np.dot(JPMD, np.log2(JPMD))

    return H2D


# Υπολογισμός της σύνθετης (διακριτής) εντροπίας τριών μεταβλητών X1, X2 και X3
# Οι πίνακες Χ1, X2 και Χ3 πρέπει να έχουν μόνο μία διάσταση
# Το αποτέλεσμα της ρουτίνας είναι η βαθμωτή τιμή της σύνθετης εντροπίας
# ο υπολογισμός της εντροπίας γίνεται με bin counting (επισφαλές για συνεχείς μεταβλητές)
def Entropy3D(X1, X2, X3):

    k = 0
    H3D = 0.0

    if np.shape(X1) == np.shape(X2) and np.shape(X1) == np.shape(X3):
        k = np.size(X1, axis=0)

        sigma1 = np.std(X1)
        bin_width1 = 3.73 * sigma1 * k ** (-1 / 5)  # Χρήση της εξίσωσης του Scott
        M1 = int((np.max(X1) - np.min(X1)) / bin_width1)
        sigma2 = np.std(X2)
        bin_width2 = 3.73 * sigma2 * k ** (-1 / 5)  # Χρήση της εξίσωσης του Scott
        M2 = int((np.max(X2) - np.min(X2)) / bin_width2)
        sigma3 = np.std(X3)
        bin_width3 = 3.73 * sigma3 * k ** (-1 / 5)  # Χρήση της εξίσωσης του Scott
        M3 = int((np.max(X3) - np.min(X3)) / bin_width3)

        frequency_joint, edges = np.histogramdd([X1, X2, X3], bins=[M1, M2, M3])
        JPMD = frequency_joint / k
        JPMD = JPMD[JPMD != 0]
        H3D = - np.dot(JPMD, np.log2(JPMD))

    return H3D


# Υπολογισμός της σύνθετης (διακριτής) εντροπίας N μεταβλητών, οι οποίες είναι οι N στήλες του πίνακα X
# Ο πίνακας Χ είναι δισδιάστατος με διαστάσεις k x N
# Το αποτέλεσμα της ρουτίνας είναι η βαθμωτή τιμή της σύνθετης εντροπίας
# ο υπολογισμός της εντροπίας γίνεται με bin counting (επισφαλές για συνεχείς μεταβλητές)
def EntropyND(X):

    k = np.size(X, axis=0)
    N = np.size(X, axis=1)
    sigma = np.std(X, axis=0)
    bin_width = 3.73 * sigma * k ** (-1 / (2+N))
    M = (np.max(X, axis=0)-np.min(X, axis=0))/bin_width
    M = M.astype(int)
    frequency_joint, edges = np.histogramdd(X, bins=M)
    JPMD = frequency_joint / k
    JPMD = JPMD[JPMD != 0]
    H = - np.dot(JPMD, np.log2(JPMD))

    return H


# Η ακόλουθη συνάρτηση υπολογίζει την σύνθετη διαφορική εντροπία τριών μεταβλητών
# Η ρουτίνα βασίζεται στον Joint probability matrix του άρθρου https://ieeexplore.ieee.org/abstract/document/6636652
# Ουσιαστικά ο κοινός πίνακας πιθανότητας γίνεται αποκτά τρεις διαστάσεις
# Δυστυχώς αυτή η ρουτίνα δουλεύει καλά μόνο για μεγάλες τιμές k λόγω της κατάρας των διαστάσεων
# Σε τρεις διαστάσεις συνήθως δεν έχω αρκετά δεδομένα για να γεμίσω τον πίνακα
def differential_entropy_3D(X, Y, Z, M):

    k = np.size(X)
    targeted_quantiles = np.linspace(0, 1 + 1 / M, M + 1, False)
    binsX = Entropy.my_percentiles(X, targeted_quantiles)
    binsY = Entropy.my_percentiles(Y, targeted_quantiles)
    binsZ = Entropy.my_percentiles(Z, targeted_quantiles)
    # binsX = percentile(X, targeted_quantiles)
    # binsY = percentile(targeted_quantiles, Y)

    frequency_joint, edges = np.histogramdd([X, Y, Z], bins=(binsX, binsY, binsZ))
    P = frequency_joint / k
    # margX = np.sum(frequency_joint, axis=0)
    # margY = np.sum(frequency_joint, axis=1)

    DX = np.zeros(M)
    DY = np.zeros(M)
    DZ = np.zeros(M)
    for i in range(0, M):
        DX[i] = binsX[i + 1] - binsX[i]
        DY[i] = binsY[i + 1] - binsY[i]
        DZ[i] = binsZ[i + 1] - binsZ[i]

    hXY = 0
    for i in range(0, M):
        for j in range(0, M):
            for k in range(0, M):
                if P[i, j, k] > 0:
                    hXY = hXY + P[i, j, k] * np.log2(P[i, j, k]/(DX[i]*DY[j]*DZ[k]))

    hXY = - hXY

    return hXY


# Ρουτίνα που ξαναελέγχει αν υπάρχει κάποια μεροληψία κατά τον υπολογισμό της κοινής πληροφορίας Ι(Χ,Υ)
# Η συνάρτηση αυτή χρησιμοποιείται αποκλειστικά από την συνάρτηση Mutual_Information_2D
def internal_bias_check(X, Y, M, L, ICA):
    # Αν η μεταβλητή bias_check είναι αληθής τότε κατασκευάζω τρεις κανονικές κατανομές δύο μεταβλητών με πίνακα
    # μεταβλητοτήτων Sigma για ρ=0, 0.5 και 0.95. Υπολογίζω την απόκλιση της μεθόδου από την θεωρητική τιμή για κάθε
    # μία από τις τρεις κανονικές κατανομές, και τις αποθηκεύω στον πίνακα partial_bias. Ο μέσος όρος των στοιχείων του
    # πίνακα αυτού είναι μία εκτιμήση της μεροληψίας της μεθόδου.

    mu = np.array([0, 0])
    ro = np.array([0, 0.5, 0.95])
    partial_bias = np.zeros(3)
    sigma1 = np.std(X)
    sigma2 = np.std(Y)
    sigma1_2 = sigma1 ** 2
    sigma2_2 = sigma2 ** 2
    N = np.size(X)

    for i in range(0, 3):
        Sigma = np.array([[sigma1_2, sigma1 * sigma2 * ro[i]], [sigma1 * sigma2 * ro[i], sigma2_2]])
        test_samples = np.random.multivariate_normal(mu, Sigma, size=N, check_valid='warn', tol=1e-8)
        hX_test = Entropy.differential_entropy_1D(test_samples[:, 0], M)
        hY_test = Entropy.differential_entropy_1D(test_samples[:, 1], M)

        hXY_test = 0
        if not ICA:
            hXY1_test = Entropy.differential_entropy_2D(test_samples[:, 0], test_samples[:, 1], L)
            hXY2_test = Entropy.differential_entropy_2D(test_samples[:, 1], test_samples[:, 0], L)
            hXY_test = (hXY1_test + hXY2_test) / 2
            if np.isnan(hXY1_test) and not np.isnan(hXY2_test):
                hXY_test = hXY2_test
            elif np.isnan(hXY2_test) and not np.isnan(hXY1_test):
                hXY_test = hXY1_test

        # Αν το hXY_test είναι nan τότε χρησιμοποιώ αναγκαστικά την συνάρτηση differential_entropy_ND
        if ICA or np.isnan(hXY_test):
            hXY_test = Entropy.differential_entropy_ND(test_samples, M)

        partial_bias[i] = hX_test + hY_test - hXY_test + 0.7213475 * math.log((1 - ro[i] ** 2))
    bias = np.average(partial_bias)
    print("Estimated bias = ", bias, " bits")

    return bias

# ########################### FAKE ###############################################################################
# Η ακόλουθη ρουτίνα δεν είναι πραγματικό κριτήριο
# χρησιμοποιείται απλώς στον υπολογισμό της διαπληροφορίας για κάθε μήνα του έτους από το ήδη βαθμονομημένο μοντέλο
def calc_transformation(observed, simulated):

    period = 12
    transinf, Hx, Hy, Hxy = Entropy.get_transinformation(observed, simulated, 20, period)

    fig1, axs1 = plt.subplots()
    fig1.suptitle('Observed vs Simulated runoff')
    axs1.plot(np.arange(0, np.size(observed)), observed, 'c', label="Observed Runoff")
    axs1.plot(np.arange(0, np.size(simulated)), simulated, 'tab:orange', label="Simulated Runoff")
    axs1.set_xlabel("[Time index]")
    axs1.set_ylabel("[mm]")
    axs1.legend(loc='upper right', shadow=False, fontsize='medium')

    fig2, axs2 = plt.subplots()

    fig2.suptitle('a-priori uncertainty vs a-posteriori uncertainty vs information gain')
    x = np.arange(0, period)
    axs2.bar((x - .2), height=Hx, width=0.2, label="a-priori uncertainty")
    axs2.bar(x, height=(Hxy-Hy), width=0.2, label="a-posteriori uncertainty")
    axs2.bar((x + .2), height=transinf, width=0.2, label="information gain")
    axs2.set_xlabel("[Month index]")
    axs2.set_ylabel("[Bits]")
    axs2.legend(loc='upper right', shadow=False, fontsize='medium')

    plt.show()
    print("")
    print("---------------------- REPORT ---------------------------")
    print("TRANSINFORMATION PER MONTH [BITS]")
    print(transinf)
    print("average transinformation=", np.average(transinf))
    print("")
    print("A-PRIORI UNCERTAINTY PER MONTH [BITS]")
    print(Hx)
    print("average a-priori uncertainty=", np.average(Hx))
    print("")
    print("A-POSTERIORI UNCERTAINTY PER MONTH [BITS]")
    print(Hxy-Hy)
    print("average a-posteriori uncertainty=", np.average(Hxy-Hy))
    print("---------------------------------------------------------")
    print("")
    return np.average(transinf)


def TR(observed, simulated):

    tr, Hx, Hy, Hxy = Entropy.get_transinformation(observed, simulated, binn=20, time_div=12)
    return np.average(tr)