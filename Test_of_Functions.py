import Entropy
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import math
import Outdated_Entropy

sigma1 = 1.054195
sigma2 = 1.615175
sigma3 = 1.5
r = .5
mu = np.array([0, 0])
population = 1000

test1 = False   # Δύο ανεξάρτητες μεταβλητές ακολουθούν την κανονική κατανομή. Πρέπει IXY=0
test2 = False   # Δύο εξαρτημένες μεταβλητές που ακολουθούν την κανονική κατανομή. Πρέπει Ι(Χ1,Χ2)=-0.5*ln(1-ρ^2)
test3 = False  # Σύγκριση των συναρτήσεων Entropy.differential_entropy_2D και Entropy.differential_entropy_2D_v2
test4 = False   # Τρεις ανεξάρτητες μεταβλητές που ακολουθούν την κανονική κατανομή. Πρέπει I(XYZ)=0
test5 = True  # Ελέγχεται αν η συνάρτηση differential_entropy_ND δίνει σωστά αποτελεσμάτα για μία κανονική κατανομή n μεταβλητών

# Test 1: Δύο ανεξάρτητες μεταβλητές ακολουθούν την κανονική κατανομή. Πρέπει IXY=0
if test1:
    Sigma = np.array([[sigma1**2, 0], [0, sigma2**2]])
    X = np.random.multivariate_normal(mu, Sigma, size=population, check_valid='warn', tol=1e-8)

    M, L = Entropy.adjustSteps(X[:, 0], X[:, 1])
    print("M = ", M, ", L = ", L)

    IXY1 = Entropy.Mutual_Information_2D(X[:, 0], X[:, 1], M, L, suppress_negatives=False, noise=4, ICA=False)

    XX = np.reshape(X[:, 0], (-1, 1))
    IXY2 = mutual_info_regression(XX, X[:, 1], discrete_features=False, n_neighbors=3, copy=True, random_state=True)
    IXY2 = IXY2[0] * 1.442695

    IXY3 = Entropy.Mutual_Information_2D(X[:, 0], X[:, 1], M, L, suppress_negatives=False, noise=4, ICA=True)

    IXY4, sigmaIXY4 = Entropy.Mutual_Information_ND(XX, X[:, 1], M, L, rep=12, noise=4, IXYmin=0)

    print("Test 1: Two independent normal variables:")
    print("Mutual_Information_2D = ", IXY1, " , sklearn = ", IXY2, " , Mutual_Information_2D_ICA = ", IXY3,
          " Mutual_Information_ND = ", IXY4, "vs theoretical value = 0.0")
    print("")


# Test 2: Δύο εξαρτημένες μεταβλητές που ακολουθούν την κανονική κατανομή. Πρέπει Ι(Χ1,Χ2)=-0.5*ln(1-ρ^2)
# όπου ρ ο συντελεστής του pearson
# Για τα παρακάτω συνθετικά δεδομένα πρέπει να προκύπτει πως H(X1,X2)=4.335 bits
# H(X1)=2.123 bits, H(X2)=2.739 bits, I(X1,X2)=0.527 bits
# Οι τιμές αυτές έχουν προκύψει από αναλυτικούς υπολογισμούς (https://en.wikipedia.org/wiki/Mutual_information)
if test2:
    Sigma = np.array([[sigma1**2, sigma1*sigma2*r], [sigma1*sigma2*r, sigma2**2]])
    X = np.random.multivariate_normal(mu, Sigma, size=population, check_valid='warn', tol=1e-8)

    M, L = Entropy.adjustSteps(X[:, 0], X[:, 1])
    print("M = ", M, ", L = ", L)

    IXY1 = Entropy.Mutual_Information_2D(X[:, 0], X[:, 1], M, L, suppress_negatives=False, noise=4, ICA=False)

    IXY2 = mutual_info_regression(np.reshape(X[:, 0], (-1, 1)), X[:, 1], discrete_features=False, n_neighbors=3, copy=True, random_state=True)
    IXY2 = IXY2[0] * 1.442695

    IXY3 = Entropy.Mutual_Information_2D(X[:, 0], X[:, 1], M, L, suppress_negatives=False, noise=4, ICA=True)

    XX = np.reshape(X[:, 0], (-1, 1))
    IXY4 = Entropy.Mutual_Information_ND(XX, X[:, 1], M, L, rep=12, noise=4, IXYmin=0)

    IXY5 = (-.5 * math.log((1-r**2)))*1.442695

    print("Test 2: Two correlated normal variables:")
    print("Mutual_Information_2D = ", IXY1, " , sklearn = ", IXY2, " , Mutual_Information_ICA = ", IXY3,
          " Mutual_Information_ND = ", IXY4, "vs theoretical value= ", IXY5)
    print("")

# Test 3: Σύγκριση των συναρτήσεων Entropy.differential_entropy_2D, Entropy.differential_entropy_2D_v2 και
# με την συνάρτηση entropy.differential_entropy_ND που χρησιμοποιεί την ανάλυση ICA
if test3:
    Sigma = np.array([[sigma1**2, sigma1*sigma2*r], [sigma1*sigma2*r, sigma2**2]])
    mu = np.array([0, 0])
    X = np.random.multivariate_normal(mu, Sigma, size=population, check_valid='warn', tol=1e-8)
    M, L = Entropy.adjustSteps(X[:, 0], X[:, 1])
    hXY1 = Entropy.differential_entropy_2D(X[:, 0], X[:, 1], L)
    hXY2 = Entropy.differential_entropy_2D_v2(X[:, 0], X[:, 1], M)
    hXY3 = Entropy.differential_entropy_2D_v2(X[:, 0], X[:, 1], L)
    hXY4 = Entropy.differential_entropy_ND(X, M, L, rep=10)

    detSigma = np.linalg.det(Sigma)
    parenthesis = detSigma*(2*math.pi*math.exp(1))**2
    hXY5 = .5 * math.log2(parenthesis)

    print("Test 3: Compare functions differential_entropy_2D vs differential_entropy_2D_v2")
    print("hXY2D = ", hXY1, " hXY2D_v2(M) = ", hXY2, " hXY2D_v2(L) = ", hXY3, " hND = ", hXY4, " h_theory=", hXY5)
    print("")

# Test 4: Τρεις ανεξάρτητες μεταβλητές που ακολουθούν την κανονική κατανομή. Πρέπει I(XYZ)=0
# Συμπέρασμα: Το IXYZ υπολογίζεται σωστά για πολύ μεγάλες τιμές του population
if test4:
    mu = np.array([0, 0, 0])
    Sigma = np.array([[sigma1**2, 0, 0], [0, sigma2**2, 0], [0, 0, sigma3**2]])
    X = np.random.multivariate_normal(mu, Sigma, size=population, check_valid='warn', tol=1e-8)
    C1 = X[:, 0]
    C2 = X[:, 1]
    C3 = X[:, 2]
    M, L = Entropy.adjustSteps(C1, C2)
    IXYZ1 = Entropy.differential_entropy_1D(C1, M)+Entropy.differential_entropy_1D(C2, M)+\
           Entropy.differential_entropy_1D(C3, M)+Outdated_Entropy.differential_entropy_3D(C1, C2, C3, int(L/1.5))-\
           Entropy.differential_entropy_2D_v2(C1, C2, L)-Entropy.differential_entropy_2D_v2(C1, C3, L)-Entropy.differential_entropy_2D_v2(C2, C3, L)

    IXYZ2 = Entropy.differential_entropy_1D(C1, M)+Entropy.differential_entropy_1D(C2, M)+\
           Entropy.differential_entropy_1D(C3, M)+Entropy.differential_entropy_ND(X, M, L, rep=10)-Entropy.differential_entropy_2D_v2(C1, C2, L)-\
            Entropy.differential_entropy_2D_v2(C1, C3, L)-Entropy.differential_entropy_2D_v2(C2, C3, L)

    IXYZ3 = Entropy.differential_entropy_1D(C1, M)+Entropy.differential_entropy_1D(C2, M)+\
           Entropy.differential_entropy_1D(C3, M)+Entropy.differential_entropy_ND(X, M, L, rep=10)-\
           Entropy.differential_entropy_ND(X[:, [0, 1]], M, L, rep=10)-Entropy.differential_entropy_ND(X[:, [1,2]], M, L, rep=10)-\
           Entropy.differential_entropy_ND(X[:, [0, 2]], M, L, rep=10)  # Η τρίτη μέθοδος δίνει το καλύτερο αποτέλεσμα

    print("Test 4: Three independent normal variables:")
    print("Mutual information calculated by Entropy.differential_entropy_3D = ", IXYZ1)
    print("Mutual information calculated by Entropy.differential_entropy_ND = ", IXYZ2)
    print("Mutual information. All entropies calculated by Entropy.differential_entropy_ND = ", IXYZ3)
    print("")

if test5:

    n = 10
    mu = np.random.uniform(10, 1000, n)

    # ------------------------ Κατασκευάζω έναν τυχαίο πίνακα μεταβλητοτήτων - συμμεταβλητοτήτων με διαστάσεις nxn ------------------------------------------------
    Sigma = np.triu(np.random.uniform(.01, 2, [n, n]))
    for i in range(0, n):
        for j in range(i + 1, n):
            Sigma[i, j] = np.random.uniform(0.01, .1) * Sigma[i, j]  # Θέλω να μειώσω τις τιμές των μη διαγώνιων στοιχείων σε σχέση με αυτές των διαγώνιων
            Sigma[j, i] = Sigma[i, j]

    # Βρίσκω την μικρότερη ιδιοτιμή η οποία δεν θέλω να είναι αρνητική, αλλιώς ο πίνακας s δεν μπορεί να είναι πίνακας μεταβλητοτήτων - συμμεταβλητοτήτων
    # Ο πίνακας sigma πρέπει να είναι συμμετρικός και θετικά ημι-ορισμένος για να είναι πίνακας μεταβλητοτήτων - συμμεταβλητοτήτων
    # https: // gowrishankar.info / blog / why - covariance - matrix - should - be - positive - semi - definite - tests - using - breast - cancer - dataset /
    min_eig = np.min(np.real(np.linalg.eigvals(Sigma)))
    if min_eig < 0:
        Sigma -= 10 * min_eig * np.eye(*Sigma.shape)
    # https: // stackoverflow.com / questions / 41515522 / numpy - positive - semi - definite - warning
    # ------------------------------------------------------------------------------------------------------------------------------------------------------

    X = np.random.multivariate_normal(mu, Sigma, size=population, check_valid='warn', tol=1e-8)
    K = np.cov(X.T)
    hjoint1 = Entropy.differential_entropy_ND(X, 30, 8, rep=5)

    detSigma = np.linalg.det(K)
    parenthesis = detSigma*(2*math.pi*math.exp(1))**n
    hjoint2 = .5 * math.log2(parenthesis)

    print("Test 5: Joint entropy of ", n, " gaussian joint variables:")
    print("calculated of joint entropy =", hjoint1)
    print("theoretical value of joint entropy =", hjoint2)
    print("relative error (%) = ", 100*(hjoint1-hjoint2)/hjoint2)
    print("")
