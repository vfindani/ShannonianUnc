import math
import os
import sys
from contextlib import contextmanager

import numpy as np
from pyDOE2 import *
from sklearn.decomposition import FastICA
from PIL import Image

import Genetic_Algorithm
import Hydrological_Models
import Criterion_Catalog


# Η ρουτίνα αυτή βρίσκει την αβεβαιότητα ενός μοντέλου για την περίοδο βαθμονόμησης
# χρησιμοποιεί όλες τις υπόλοιπες ρουτίνες του module
def diagnose_model(model, limits, basin_info, criterion, cdates, vdates):

    # ********************************* ΒΗΜΑ 1: ΒΑΘΜΟΝΟΜΕΙΤΑΙ ΤΟ ΜΟΝΤΕΛΟ ****************************************
    # και προκύπτει το βέλτιστο σετ παραμέτρων
    print("STEP 1: Calibrating model ...")
    best_vector, ccrit = Genetic_Algorithm.Genetic_Algorithm(limits, model, criterion, cdates, basin_info, 200, 0.001)
    best_vector = np.resize(best_vector, (1, np.size(best_vector)))
    print("STEP 1: Model was Calibrated!")

    # ********************************* ΒΗΜΑ 2: ΒΡΙΣΚΩ ΤΟ QSIM **************************************************
    # Το μοντέλο τρέχει για το χρονικό διάστημα του πίνακα cdates ώστε να προκύψει η χρονοσειρά των προσομοιωμένων
    # απορροών. Η εκτέλεση του μοντέλου γίνεται με το βέλτιστο σετ παραμέτρων
    print("STEP 2: Finding the Qsim for the best parametric set ...")
    model0, X = Hydrological_Models.unify_data(model, cdates, basin_info)
    cQsim = getattr(Hydrological_Models, model0)(best_vector, 1, np.array([["just_return_Qsim"], [1]]), X, np.array([[0]], dtype=object))[0][0]
    print("STEP 2: Qsim was found!")

    # ******************************* ΒΗΜΑ 3: ΠΡΟΕΤΟΙΜΑΖΩ ΤΟΥΣ ΠΙΝΑΚΕΣ Xobs, Qobs και Qsim ****************
    cQsim_no_nans = cQsim[~np.isnan(X[:, - 1])].T  # Το cQsim_no_nans είναι το cQsim χωρίς τις τιμές που αντιστοιχούν στις κενές καταγραφές απορροής
    X = X[~np.isnan(X[:, - 1])]  # Από τον πίνακα Χ αφαιρώ τις γραμμές που έχουν κενές καταγραφές απορροής
    Xobs = X[:, 1:-1]  # Στο Xobs δεν συμπεριλαμβάνεται η 1η στήλη του πίνακα Χ που είναι η εξατμισοδιαπνοή γιατί συνήθως υπολογίζεται από τα P και Τ.
    Qobs = X[:, - 1].T

    # *********************** ΒΗΜΑ 4: ΕΠΑΛΗΘΕΥΣΗ ΤΟΥ ΜΟΝΤΕΛΟΥ ΓΙΑ ΤΗΝ ΠΕΡΙΟΔΟ VALIDATION DATES *******************
    print("STEP 3: Validating model ...")
    model0, X = Hydrological_Models.unify_data(model, vdates, basin_info)
    # Χρειάζομαι το υδρογράφημα για την περίοδο επαλήθευσης αλλά και το κριτήριο για την περίοδο επαλήθευσης
    vQsim = getattr(Hydrological_Models, model0)(best_vector, 1, np.array([["just_return_Qsim"], [1]]), X, np.array([[0]], dtype=object))[0][0]
    vcrit = getattr(Criterion_Catalog, str(criterion[0][0]))(X[:, - 1], vQsim)
    print("STEP 3: Model was Validated!")

    # ******************************* ΒΗΜΑ 5: ΥΠΟΛΟΓΙΖΩ ΤΗΝ ΜΕΓΙΣΤΗ ΕΠΕΞΗΓΗΣΙΜΗ ΠΛΗΡΟΦΟΡΙΑ ΣΕ BITS **************** (για την περίοδο βαθμονόμησης cdates)
    print("STEP 4: Calculating Capacity ...")
    zero_theta_point, ExpInfMax = Genetic_Algorithm.Genetic_Algorithm(limits, model, np.array([["ExpInfLog"], [1]]), cdates, basin_info, 200, 0.01)
    print("STEP 4: Capacity was Calculated!")

    # https://www.youtube.com/watch?v=EoV4seyWs8k&ab_channel=Nightwish-Topic
    # ******************************* ΒΗΜΑ 6: ΥΠΟΛΟΓΙΖΩ ΤΑ ReqInf, AvInf και ExpInf ΣΕ BITS ********************** (για την περίοδο βαθμονόμησης)
    print("STEP 5: Calculating Uncertainty Components ...")
    ReqInf, AvInf, ExpInf, ExpInfMax, SigmaAvInf = Aleatory_n_Epistemic_Uncert(Xobs, Qobs, cQsim_no_nans, C=6, Dx=10, ExpInfMax=ExpInfMax)

    # ******************************* ΒΗΜΑ 7: ΥΠΟΛΟΓΙΖΩ ΤΑ Φ, Ψ, Θ ΚΑΙ Δ ****************************************
    phi = ReqInf - AvInf
    psi = AvInf - ExpInf
    delta = AvInf - ExpInfMax
    theta = ExpInfMax - ExpInf
    print("STEP 5: Uncertainty Components were Calculated.")

    # γράφω τα αποτελέσματα στο αρχείο
    with open('Results_Uncertainty_Analysis.txt', 'a') as the_file:
        line_string = str(ReqInf) + "," + str(AvInf) + "," + str(ExpInf) + "," + str(ExpInfMax) + "," + str(SigmaAvInf) + "," + str(phi) + ","\
                      + str(psi) + "," + str(delta) + "," + str(theta) + "," + str(ccrit-9) + "," + str(vcrit-9) + "\n"
        the_file.write(line_string)

    # τυπώνω τα προσομοιωμένα υδρογραφήματα για τις περιόδους βαθμονόμησης και επαλήθευσης στο αρχείο
    with open('Computed_Hydrographs.txt', 'a') as the_file:
        the_file.write(str(model) + "," + str(basin_info[0]) + "\n")
        the_file.write("calibrated parameters =" + str(np.stack(best_vector)[0]) + "\n")
        the_file.write("zero theta point =" + str(np.stack(zero_theta_point.T)[0]) + "\n")
        the_file.write("Computed_Hydrograph_for_calibration_period:" + str(cdates) + "\n")
        for i in range(0, np.size(cQsim)):
            the_file.write(str(i) + "," + str(cQsim[i]) + "\n")
        the_file.write("Computed_Hydrograph_for_validation_period:" + str(vdates) + "\n")
        for i in range(0, np.size(vQsim)):
            the_file.write(str(i) + "," + str(vQsim[i]) + "\n")


# Η ρουτίνα αυτή υπολογίζει τις αβεβαιότητες σε bits
# Το C είναι το βήμα έως το οποίο θα μετατοπιστούν οι χρονοσειρές των δεδομένων εισόδου
# Πχ αν χρονοσειρές εισόδου είναι οι R(t) και PET(t) και επιλέξω C=2 τότε θα δημιουργηθούν οι
# χρονοσειρές R(t-1), PET(t-1), R(t-2), PET(t-2) και θα εννωθούν όλες μαζί στον πίνακα X1
def Aleatory_n_Epistemic_Uncert(X, Qobs, Qsim, C, Dx, ExpInfMax):

    # Προεργασία των δεδομένων. Κόβω τις αρνητικές και μηδενικές τιμές.
    X[X <= 0] = 0.01
    Qobs[Qobs <= 0] = 0.01
    Qsim[Qsim <= 0] = 0.01

    # !!!!!!!!!!!!!!!!!!!! Υπολογισμός της απαιτούμενης πληροφορίας !!!!!!!!!!!!!!!!!!!!
    # ή εναλλακτικά ReqInf = discrete_entropy1D(Qobs, Dx)[0]
    ReqInf = differential_entropy_1D(Qobs, M=20) - np.log2(Dx)

    # Για τον υπολογισμό της διαθέσιμης και της επεξηγήσιμης πληροφορίας αλλάζω μεταβλητές. Έτσι
    # παίρνω τον λογάριθμο των μεταβλητών. Η κοινή πληροφορία δεν μεταβάλλεται από την αλλαγή των μεταβλητών
    # βλέπε το άρθρο DOI: 10.1103/PhysRevE.69.066138 .  Για κάποιο λόγο η μέθοδος ICA δουλεύει καλά όταν οι
    # μεταβλητές ακολουθούν στο περίπου την κανονική κατανομή. Έτσι αφού τα X, Qobs και Qsim ακολουθούν στο
    # περίπου την λογαριθμοκανονική κατανομή, οι λογάριθμοι τους ακολουθούν την κανονική κατανομή.
    X = np.log(X)
    Qobs = np.log(Qobs)
    Qsim = np.log(Qsim)

    # !!!!!!!!!!!!!!!!!!!! Υπολογισμός της επεξηγήσιμης πληροφορίας !!!!!!!!!!!!!!!!!!!!
    print("... Calculating ExpInf ...")
    M, L = adjustSteps(Qobs, Qsim)
    ExpInf = Mutual_Information_2D(Qobs, Qsim, M, L, suppress_negatives=True, noise=3, ICA=False)

    # Πρέπει πάντα η μέγιστη επεξεγίσιμη πληροφορία να είναι μεγαλύτερη από την οποιαδήποτε τιμή εξ ορισμού.
    if ExpInfMax < ExpInf:
        ExpInfMax = ExpInf
    print("... ExpInf Calculated ...")

    # Δημιουργούνται οι υστερημένες χρονοσειρές αν C>0
    k = np.size(X, axis=0)
    X1 = X
    if C > 0:
        Xlagged = np.zeros((k, np.size(X, axis=1), C))  # 3Δ Πίνακας διαστάσεων k, np.size(X,axis=1) και C
        for j in range(0, C):
            for i in range(0, k-j-1):
                Xlagged[i+j+1, :, j] = X[i, :]
            X1 = np.hstack((X1, Xlagged[:, :, j]))

    # !!!!!!!!!!!!!!!!!!!! Υπολογισμός της διαθέσιμης πληροφορίας !!!!!!!!!!!!!!!!!!!!
    # Δεν χρησιμοποιώ τις πρώτες C σειρές των πινάκων X1 και Qobs γιατί περιέχουν μηδενικά στοιχεία
    # λόγω της μετατόπισης των υστερημένων χρονοσειρών
    print("... Calculating AvInf ...")
    AvInf, SigmaAvInf = Mutual_Information_ND(X1[C:, :], Qobs[C:], M, L=None, rep=20, noise=3, IXYmin=0.0)
    print("... AvInf Calculated ...")
    # Σύμφωνα με τον Gupta (βλέπε στο 43:32 του βίντεο
    # https://www.youtube.com/watch?v=OB8A9sT8XY8&t=701s&ab_channel=InformationTheoryintheGeosciences
    # πρέπει ο αριθμός των ποσοστημορίων να είναι ίσος με το 25% του πλήθους του δείγματος (k)

    return ReqInf, AvInf, ExpInf, ExpInfMax, SigmaAvInf


# Υπολογίζει την διαφορική εντροπία για μία μεταβλητή x
# M είναι ο αριθμός των ποσοστομορίων
def differential_entropy_1D(X, M):

    targeted_quantiles = np.linspace(0, 1+1/M, M+1, False)
    # bins = percentile(targeted_quantiles, X)
    bins = my_percentiles(X, targeted_quantiles)
    DX = np.zeros(M)

    for i in range(0, M):
        DX[i] = bins[i + 1] - bins[i]
        if DX[i] == 0:
            print("zero bin found")

    hX = np.log2(M) + (1/M)*np.sum(np.log2(DX))

    return hX


# Η ρουτίνα αυτή είναι η κύρια ρουτίνα υπολογισμού της σύνθετης διαφορικής πληροφορίας 2 μεταβλητών
# βασίζεται στην κατασκευή ισοπίθανων κλάσεων με την χρήση δισδιάστατων ποσοστημόριων
# ουσιαστικά χρησιμοποιεί το Adaptive grid του άρθρου https://ieeexplore.ieee.org/abstract/document/6636652
def differential_entropy_2D(X, Y, M):
    # https://www.youtube.com/watch?v=h4SI9A3Yh20
    k = np.size(X)
    targeted_quantiles = np.linspace(0, 1 + 1 / M, M + 1, False)
    # binsX είναι ο πίνακας που περιέχει τα όρια των κλάσεων κατά x. Αφού έχω Μ κλάσεις κατά Χ,
    # η διάσταση του πίνακα binsX είναι Μ+1
    binsX = my_percentiles(X, targeted_quantiles)

    # binsy είναι ο πίνακας που περιέχει τα όρια των κλάσεων κατά y. Τα όρια των κλάσεων μεταβάλλονται συναρτήσει της
    # κλάσης του Χ. Ο 1ος δείκτης αναφέρεται στην x-κλάση και για αυτό η 1η διάσταση έχει μήκος Μ. Ο 2ος δείκτης
    # αναφέρεαι τα όρια των y-κλάσεων και για αυτό έχει μήκος Μ+1 (αφού οι y-κλάσεις είναι Μ σε αριθμό)
    binsY = np.zeros((M, M+1))
    binsX_counter = np.zeros(M, dtype=int)
    Y_arranged = np.empty((M, 3*int(k/M)))  # Δίνω μεγάλο μήκος στην 2η διάσταση καθώς δεν ξέρω apriori το μέγεθος της
    Y_arranged[:] = np.nan

    DX = np.zeros(M)
    DY = np.zeros((M, M))

    # Ελέγχω αν το t-οστό σημείο (X[t], Y[t]) ανιστοιχεί στην κλάση i του άξονα x
    # Η διαδικασία αυτή γίνεται για κάθε σημείο t και για κάθε x-κλάση i
    for i in range(0, M):
        for t in range(0, k):
            if binsX[i] <= X[t] < binsX[i+1]:  # Αν το σημείο t βρίσκεται στην κλάση i κατά x ...
                binsX_counter[i] += 1  # ... αυξάνω των αριθμό των σημείων στην κλάση αυτή κατά 1, και ...
                Y_arranged[i, binsX_counter[i]] = Y[t]
                # ... βάζω το Υ του σημείου στον πίνακα Υ_arranged απ' όπου θα προκύψουν τα ποσοστημόρια

        # Αφού τελειώσω την κατάταξη των σημείων (X[t], Y[t]) στην i-οστή κλάση βρίσκω τα εκατοστημόρια της
        # Η γραμμή i του πίνακα Y_arranged (πίνακας Z) αντιστοιχεί στην δεσμευμένη κατανομή πιθανοτήτων P(Y|X)
        # διότι αφορά δεσμευμένες τιμές του Χ
        Z = Y_arranged[i, :]
        Z = Z[~np.isnan(Z)]  # Αφαιρώ τα στοιχεία Nan
        binsY[i, :] = my_percentiles(Z, targeted_quantiles)

        # Βρίσκω τα πλάτη των κλάσεων
        DX[i] = binsX[i+1]-binsX[i]

        for j in range(0, M):
            DY[i, j] = binsY[i, j+1] - binsY[i, j]

    # Παράγεται το ιστόγραμμα των μεταβλητών Χ και Υ για έλεγχο και υπολογισμό του δείκτη ομοιομορφίας.
    # Θα πρέπει όλες οι κλάσεις να έχουν μέσα τους περίπου τον ίδιο αριθμό σημείων
    histogram = np.zeros((M, M))
    for t in range(0, k):
        for i in range(0, M):
            for j in range(0, M):
                if binsX[i] <= X[t] < binsX[i + 1] and binsY[i, j] <= Y[t] < binsY[i, j + 1]:
                    histogram[i, j] += 1

    # Υπολογίζω τον δείκτη ομοιομορφίας του ιστογράμματος
    uniformity = np.rint(np.min(histogram)/np.max(histogram))
    # print("The derived histogram is the following:")
    # print(histogram)
    # print("and it has a uniformity equal to ", uniformity)

    # Aν ο δείκτης ομοιομορφίας του ιστογράμματος είναι ίσος με 1, τότε τα σημεία έχουν κατανεμηθεί ομοιόμορφα
    # και μπορούμε να χρησιμοποιήθουμε τον τύπο για τις ισοπίθανες κλάσεις...
    hXY = 0
    if uniformity == 1:
        P = 1 / (M ** 2)
        for i in range(0, M):
            for j in range(0, M):
                hXY = hXY + P * np.log2(P/(DX[i]*DY[i, j]))
        hXY = - hXY
    else:
        # ... αλλιώς η τιμή της κοινής εντροπίας δεν μπορεί να υπολογιστεί από την μέθοδο των ισοπίθανων κλάσεων
        hXY = np.nan

    return hXY


# Η συνάρτηση αυτή είναι εναλλακτική της differential_entropy_2D
# και έχει ελαφρώς χαμηλότερη ακρίβεια. Οι τιμές που δίνει είναι σχετικά καλές
# Γενικότερα είναι πιο απλή από την βασική εκδοχή και είναι πιο σταθερή υπολογιστικά
# Έτσι όταν η βασική συνάρτηση δίνει nan, χρησιμοποιείται αυτή
# Η ρουτίνα βασίζεται στον Joint probability matrix του άρθρου https://ieeexplore.ieee.org/abstract/document/6636652
def differential_entropy_2D_v2(X, Y, M):

    k = np.size(X)
    targeted_quantiles = np.linspace(0, 1 + 1 / M, M + 1, False)
    binsX = my_percentiles(X, targeted_quantiles)
    binsY = my_percentiles(Y, targeted_quantiles)
    # binsX = percentile(X, targeted_quantiles)
    # binsY = percentile(targeted_quantiles, Y)

    frequency_joint, bins_obs, bins_sim = np.histogram2d(X, Y, bins=(binsX, binsY))
    P = frequency_joint / k
    # margX = np.sum(frequency_joint, axis=0)
    # margY = np.sum(frequency_joint, axis=1)

    DX = np.zeros(M)
    DY = np.zeros(M)
    for i in range(0, M):
        DX[i] = binsX[i + 1] - binsX[i]
        DY[i] = binsY[i + 1] - binsY[i]

    hXY = 0
    for i in range(0, M):
        for j in range(0, M):
            if P[i, j] > 0:
                hXY = hXY + P[i, j] * np.log2(P[i, j]/(DX[i]*DY[j]))

    hXY = - hXY

    return hXY


# Η ρουτίνα αυτή υπολογίζει την σύνθετη εντροπία Ν μεταβλητών χρησιμοποιώντας της μέθοδο ICA
# δηλαδή χωρίζει τον πίνακα Χ στις ανεξάρτητες συνιστώσες S και υπολογίζει την εντροπία του
# Χ, ως h(x) = h(S) + log(abs(det(A))).
def differential_entropy_ND(X, M, L, rep):

    n = np.size(X, axis=1)

    if L is None or L == 0 or rep == 1:  # Κάνω μόνο μία επανάληψη αν το L δεν δίνεται ή είναι μηδέν ή rep=1
        ica = FastICA(n_components=n, whiten='unit-variance', max_iter=400, tol=1e-8, algorithm='deflation', whiten_solver='svd', random_state=None)
        Sources = ica.fit_transform(X)  # <---------
        A = ica.mixing_
        hND = np.log2(np.abs(np.linalg.det(A)))  # Το hND έχει μόνο μία τιμή
        for i in range(0, n):
            hND += differential_entropy_1D(Sources[:, i], M)
        return hND

    elif rep > 1:  # Αυτό γίνεται στο άρθρο (Gong et al. 2013). Το L χρησιμοποιείται μόνο στην περίπτωση που rep>1.
        hND = np.zeros(rep)  # To hND είναι πίνακας με rep τιμές
        criterion_of_best = np.zeros(rep)
        for q in range(0, rep):
            ica = FastICA(n_components=n, whiten='unit-variance', max_iter=400, tol=1e-8, algorithm='deflation', whiten_solver='svd', random_state=None)
            Sources = ica.fit_transform(X)  # Βρίσκω τις ανεξάρτητες συνιστώσες των σημάτων Χ
            A = ica.mixing_
            hND[q] = np.log2(np.abs(np.linalg.det(A)))
            for i in range(0, n):
                hND[q] += differential_entropy_1D(Sources[:, i], M)

            # Αφού κάνω την παραπάνω διαδικασία rep φορές πρέπει κάπως να δω ποια τιμή του hND από τις rep θα κρατήσω. Έτσι υπολογίζω για κάθε ζεύγος σημάτων
            # την 2Δ κοινή τους πληροφορία και στην συνέχεια για την κάθε περίπτωση rep, αθροίζω τα τετράγωνα των M2D. Το τρέξιμο με το μικρότερο άθροισμα τετραγώγων
            # είναι αυτό που δίνει την τελική τιμή του hND. Οι αρνητικές τιμές ρ γίνονται θετικές με την ύψωση στο τετράγωνο, και άρα μη αποδεκτές.
            M2D_between_Sources = np.zeros((n, n))  # Για κάθε επανάληψη q ο πίνακας M2D_between_Sources πρέπει να αρχικοποιείται
            for i in range(0, n-1):
                for j in range(i+1, n):
                    M2D_between_Sources[i, j] = Mutual_Information_2D(Sources[:, i], Sources[:, j], M, L, suppress_negatives=False, noise=0, ICA=False)
            criterion_of_best[q] = np.sum(np.square(M2D_between_Sources))  # Υπολογίζεται το κριτήριο για την q επανάληψη
        best_hND = np.average(hND[np.where(criterion_of_best == np.amin(criterion_of_best))])
        # Το τελικό hnD είναι αυτό το οποίο αντιστοιχεί στην μικρότερη τιμή M2D_rep. Αν έχω πολλές τιμές με την ίδια τιμή criterion_of_best επιστρέφω τον μέσο όρο τους.
        return best_hND

# Η λογική της ανωτέρω ρουτίνας:
# https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e
# Στην συνέχεια θέλω να βρω τον πίνακα Α που ικανοποιεί την σχέση X = S * A  ==> A = (S^-1) * X  ==> A = pseudoinverse(S) * X
# Ο πίνακας Α όταν πολλαπλασιαστεί με τον πίνακα S των ανεξάρτητων σημάτων δίνει τον αρχικό πίνακα X. Οι πίνακες X και S έχουν διαστάσεις k x n
# ενώ ο πίνακας A έχει διαστάσεις n x n. Το n είναι ο αριθμός των ανεξάρτητων και των πραγματικών χρονοσειρών. k είναι το χρονικό μήκος των χρονοσειρών
# και των σημάτων. Αν S και x είναι δύο πίνακες που συνδέονται με έναν γραμμικό μετασχηματισμό transpose(x) = A * transpose(S) <==> x = S * transpose(A), τότε
# h(x) = h(S) + log(abs(det(A)))  (βλέπε βιβλιο Thomas σελ 254 σχέση 8.71). Τα σήματα S είναι ανεξάρτητα, άρα h(S)=Σ h(Si). Επίσης ισχύει h(x) = h(X-μ) = h(X)
# Αν όλα έχουν πάει καλά, κατά τον υπολογισμό του Α, τότε πρέπει x - S * transpose(A) = 0 όπου x είναι ο πίνακας X μετά το κεντράρισμα του, ήτοι x = Center(X) ==> x = X - μ
# όπου μ ο πίνακας με τους μέσους όρους των στηλών του X. Άρα εν τέλει πρέπει
# X - μ - S * transpose(A) = 0
# errors = X - np.matmul(Sources, A.T) - ica.mean_  # <-------
# print("ICA residuals = ", np.average(errors, axis=0))  # <-------
# https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py


# Η βασική ρουτίνα υπολογισμού της κοινής πληροφορίας δύο μεταβλητών
# είναι εναλλακτική της ρουτίνας sklearn.feature_selection.mutual_info_regression
# Αν η μεταβλητή ICA είναι False, θα προτιμηθεί η 2D εντροπία να υπολογισθεί με την μέθοδο των ποσοστημόριων.
# Αλλιώς αν ICA = True τότε χρησιμοποείται μόνο η μέθοδος ICA και δεν έχει σημασία η τιμή του L
def Mutual_Information_2D(X, Y, M, L, suppress_negatives, noise, ICA):

    # !!!!!!!!!!!!!!! ΜΕΡΟΣ 1: ΠΡΟΣΘΗΚΗ ΤΥΧΑΙΟΥ ΘΟΡΥΒΟΥ !!!!!!!!!!!!!!!!!!!!!!!!!!
    # Προσθήκη τυχαίου θορύβου στα δεδομένα ώστε να μην έχω πολλές ίδιες τιμές
    if noise > 0:
        mag = 10 ** -noise
        q = np.size(X)
        X = X + np.random.normal(0, mag, q)
        Y = Y + np.random.normal(0, mag, q)

    # !!!!!!!!!!!!!!! ΜΕΡΟΣ 2: ΥΠΟΛΟΓΙΣΜΟΣ ΤΩΝ ΔΙΑΦΟΡΙΚΩΝ ΕΝΤΡΟΠΙΩΝ !!!!!!!!!!!!!!!!!!!!!!!!!!
    hX = differential_entropy_1D(X, M)
    hY = differential_entropy_1D(Y, M)

    # Παρακάτω γίνεται επιλογή της μεθόδου υπολογισμού της σύνθετης δισδιάστατης εντροπίας
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    hXY = 0.0
    if not ICA:  # Αν δεν έχει επιλεχθεί να δουλέψει η μέθοδος ICA, υπολογίζονται οι δισδιάστατες εντροπίες με
        # την συνάρτηση differential_entropy_2D
        hXY1 = differential_entropy_2D(X, Y, L)
        hXY2 = differential_entropy_2D(Y, X, L)

        if not np.isnan(hXY1) and not np.isnan(hXY2):
            hXY = (hXY1 + hXY2) / 2  # Αν καμία από τις τιμές δεν είναι nan, τότε η σύνθετη εντροπία είναι ο μ.όρος τους
        elif np.isnan(hXY1) and not np.isnan(hXY2):
            hXY = hXY2
            print("h(X,Y) was rejected.")
        elif np.isnan(hXY2) and not np.isnan(hXY1):
            hXY = hXY1
            print("h(Y,X) was rejected.")
        elif np.isnan(hXY1) and np.isnan(hXY2):
            # Αν δεν επιστράφηκε καμία τιμή από την μέθοδο με τα ποσοστημόρια, τότε η δισδιάστατη εντροπία υπολογίζεται με την συνάρτηση differential_entropy_2D_v2
            hXY = differential_entropy_2D_v2(X, Y, L)
            print("Both h(X,Y) and h(Y,X) were rejected. Applying the alternative H2D function... hXY= ", hXY)
    else:
        # Αν έχει επιλεχθεί να δουλέψει η μέθοδος ICA, τότε η δισδιάστατη εντροπία υπολογίζεται αποκλειστικά με την συνάρτηση differential_entropy_ND
        hXY = differential_entropy_ND((np.vstack((X, Y))).T, M, L=None, rep=1)  # Αν rep=1, το L δεν χρησιμοποιείται πουθενά εντός της συνάρτησης, το δίνω για λόγους πληρότητας

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # !!!!!!!!!!!!!!! ΜΕΡΟΣ 3: ΥΠΟΛΟΓΙΖΩ ΤΗΝ ΚΟΙΝΗ ΠΛΗΡΟΦΟΡΙΑ !!!!!!!!!!!!!!!!!!!!!!!!!!
    # Αφού έχω υπολογίσει τις διαφορικές εντροπίες hX, hY, hXY και το bias, υπολογίζω την κοινή πληροφορία
    IXY = hX + hY - hXY

    # !!!!!!!!!!!!!!! ΜΕΡΟΣ 4: ΑΠΟΚΟΠΤΩ ΤΙΣ ΑΡΝΗΤΙΚΕΣ ΚΑΙ ΜΗ ΕΠΙΤΡΕΠΤΕΣ ΤΙΜΕΣ !!!!!!!!!!!!!!!!!!!!!!!!!!
    # Γνωρίζω πως πάντα I(X,Y)>=0. Άρα αν suppress_negatives=True, μετατρέπω τις αρνητικές τιμές σε μηδενικές
    # Επίσης, πάντα πρέπει να ισχύει ο περιορισμός της σχέσης Α5 του άρθρου DOI: 10.1103/PhysRevE.69.066138
    IXYmin = MI_gaussian((np.vstack((X, Y))).T)
    if suppress_negatives is True and IXY < IXYmin:
        print("The value of the mutual information is lower than its lowest value:", IXYmin)
        IXY = IXYmin

    return IXY


# Υπολογισμός της κοινής πληροφορίας Ι(Χ;Y) υπολογίζοντας την πολυδιάστατη σύνθετη εντροπία με την μέθοδο ICA. Ο πίνακας Y πρέπει να έχει μόνο μία διάσταση.
def Mutual_Information_ND(X, Y, M, L, rep, noise, IXYmin):

    # Προσθήκη τυχαίου θορύβου στα δεδομένα ώστε να μην έχω πολλές ίδιες τιμές
    if noise > 0:
        mag = 10 ** -noise
        X = X + np.random.normal(0, mag, np.shape(X))
        Y = Y + np.random.normal(0, mag, np.shape(Y))

    k = np.size(X, axis=0)
    XY = np.hstack((X, np.resize(Y, (k, 1))))
    IXY_partial = np.zeros(rep)

    # IXYmin = max(IXYmin, MI_gaussian(XY))  # Βρίσκω την κατώτερη δυνατή τιμή του IXY βάσει του περιορισμού της σχέσης Α5 του άρθρου DOI: 10.1103/PhysRevE.69.066138
    hND_x_max = hND_gaussian(X)
    hND_xy_max = hND_gaussian(XY)
    h_y = differential_entropy_1D(Y, M)

    for i in range(0, rep):  # Ο υπολογισμός των HND γίνεται rep φορές
        hND_x = differential_entropy_ND(X, M, L, rep)
        hND_xy = differential_entropy_ND(XY, M, L, rep)
        IXY_partial[i] = hND_x + h_y - hND_xy  # Υπολογίζω την κοινή πληροφορία για την επανάληψη i

        # Αν το IXY, το hND_x ή το hND_xy έχει αναληθή τιμή (με ένα περιθώριο 5%) τότε βάζω το IXY_partial ως nan
        if IXY_partial[i] < IXYmin or hND_x > 1.05 * hND_x_max or hND_xy > 1.05 * hND_xy_max:
            IXY_partial[i] = np.nan
        print(IXY_partial[i], h_y, hND_x, hND_xy, hND_x_max, hND_xy_max)

    IXY_partial = IXY_partial[~np.isnan(IXY_partial)]  # Αφαιρώ τις nan τιμές από τον πίνακα, και μένω μόνο με τις επιτρεπτές τιμές
    IXY = np.average(IXY_partial)  # Η κοινή πληροφορία προκύπτει ως ο μέσος όρος των τιμών του πίνακα IXY_partial
    sigma = np.std(IXY_partial)

    print("")
    print("Standard deviation of ND Mutual Information = ", sigma)
    print("Mutual Information ND = ", IXY, " bits.")
    print("")

    return IXY, sigma


# Ρουτίνα για να βρίσκει τον κατάλληλο αριθμό των βημάτων Μ και L
def adjustSteps(X, Y):

    print("*******************************************************************")
    print("Searching for the best combination of M and L based on the given variables ...")

    N = np.size(X)
    sigma1 = np.std(X)
    sigma2 = np.std(Y)
    sigma1_2 = sigma1**2
    sigma2_2 = sigma2**2
    mu = np.array([0, 0])
    bias = np.zeros((7, 15))

    M = np.arange(10, 50, 5, dtype=int)
    L = np.arange(5, 20, 1, dtype=int)

    p = 1
    if 500 < N <= 1000:
        p = 2
    elif 50 < N <= 500:
        p = 4
    elif N <= 50:
        p = 6

    with suppress_stdout():
        for i in range(0, 7):
            for j in range(0, 15):
                if M[i]/p >= L[j]:
                    for ro in range(0, 100, 20):
                        Sigma = np.array([[sigma1_2, sigma1*sigma2*ro/100], [sigma1*sigma2*ro/100, sigma2_2]])
                        samples = np.random.multivariate_normal(mu, Sigma, size=N, check_valid='warn', tol=1e-8)
                        I_estm = Mutual_Information_2D(samples[:, 0], samples[:, 1], M[i], L[j], suppress_negatives=False, noise=0, ICA=False)
                        I_corr = -0.7213475 * math.log((1 - (ro/100) ** 2))
                        bias[i, j] = bias[i, j] + (I_estm-I_corr)**2
                else:
                    bias[i, j] = 100

    indices_of_min = np.array(np.where(bias == np.amin(bias)))

    best_M = int(M[indices_of_min[0]])
    best_L = int(L[indices_of_min[1]])

    print("The quest has ended. Best M = ", best_M, " and best L = ", best_L)
    print("*******************************************************************")
    print("")

    return int(M[indices_of_min[0]]), int(L[indices_of_min[1]])


# Η ρουτίνα αυτή υπολογίζει την (διακριτή) εντροπία ενός πίνακα nxk του οποίου οι στήλες είναι n χρονοσειρές
# η καθεμία από τις οποίες έχει k στοιχεία. Μπορεί να ισχύει n=1. Ο πίνακας X πρέπει να έχει 2 διαστάσεις.
# Το αποτέλεσμα της ρουτίνας αυτής είναι ένας πίνακας Η με n στοιχεία. Κάθε στοιχείο αντιπροσωπεύει την εντροπία
# μίας από τις n χρονοσειρές.
# ο υπολογισμός της εντροπίας γίνεται με bin counting (επισφαλές για συνεχείς μεταβλητές)
def discrete_entropy1D(X, bin_width):

    # Αν ο πίνακας X έχει μόνο μία διάσταση, δηλαδή είναι διάνυσμα, τότε γίνεται δισδιάστατος
    if np.ndim(X) == 1:
        X = np.reshape(X, (np.size(X, axis=0), 1))

    k = np.size(X, axis=0)
    n = np.size(X, axis=1)

    H = np.zeros(n, dtype=float)
    M = np.zeros(n, dtype=int)

    for i in range(0, n):
        M[i] = int((np.max(X[:, i])-np.min(X[:, i])) / bin_width)
        frequency_Xi, binns = np.histogram(X[:, i], bins=M[i])
        frequency_Xi = frequency_Xi / k
        frequency_Xi = frequency_Xi[frequency_Xi != 0]
        H[i] = - np.dot(frequency_Xi, np.log2(frequency_Xi))

    return H


# Αυτή η υπορουτίνα βρίσκει τα ποσοστομορία F της χρονοσειράς X
# και είναι εναλλακτική ρουτίνα του np.quantiles
def my_percentiles(X, F):

    N = np.size(F)
    X = np.array(sorted(X))
    index = F * len(X) - .5
    X0 = np.zeros(N)

    if np.size(X) > N:
        for i in range(0, N-1):
            X0[i] = X[int(index[i])]+(index[i]-int(index[i]))*(X[1+int(index[i])]-X[int(index[i])])
        X0[0] = np.min(X)
        X0[N - 1] = np.max(X)
    elif 1 < np.size(X) <= N:
        X0[0] = np.min(X)
        X0[N - 1] = np.max(X)
    elif np.size(X) == 1:
        X0[0] = X[0]-.01
        X0[N - 1] = X[0]+.01
    else:  # Αν τα στοιχεία του πίνακα Χ είναι λιγότερα από τον αριθμό των ζητούμενων ποσοστομορίων
        # γεμίζω το 1ο και τελευταίο στοιχείο του X0 με έναν αριθμό πολύ κοντά στο Χ
        # έτσι παράγω ψευδή ποσοστημόρια για να μην κρασάρει ο αλγόριθμος
        X0[0] = 0
        X0[N - 1] = 0.01

    # Αν υπάρχουν κλάσεις με μηδενικό πλάτος, το άνω όριο της κλάσης γίνεται ίσο με NaN
    # show_final = False
    DX = np.zeros(N-1)
    for i in range(0, N-2):
        DX[i] = X0[i+1]-X0[i]

    for i in range(0, N-2):
        if DX[i] == 0:
            # print("null bin found in array=", X0)
            print("null bin found.")
            if i != 0:
                X0[i] = np.nan
            X0[i+1] = np.nan
            # show_final = True

    # Οι κενές τιμές του πίνακα X0 γεμίζουν με γραμμική παρεμβολή
    nans, c = nan_helper(X0)
    X0[nans] = np.interp(c(nans), c(~nans), X0[~nans])

    # if show_final:
    #     print()
    #     # print("samples array = ", X)
    #     # print("corrected percentiles = ", X0)
    #     # print("proposed numpy percentiles = ", np.quantile(X, F))
    #     #print("################################################")

    return X0


# https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


# Συνάρτηση που βρίσκει την χαμηλότερη δυνατή τιμή της πολυδιάστατης κοινής πληροφορίας Ι(Χ1, Χ2,...ΧΝ)
def MI_gaussian(X):

    n = np.size(X, axis=1)
    Imin = 0.0
    if n >= 2:
        C = np.cov(X.T)
        Num = np.linalg.det(C)
        Denom = (np.prod(np.diag(C)))**.5
        # Αν το όρισμα του λογαρίθμου είναι μεγαλύτερο από 1 (δηλαδή όταν δίνει Imin>0), εφαρμόζεται η σχέση Α.5.
        if Num > Denom:
            Imin = .7213475 * math.log(Num / Denom)
    return Imin


def hND_gaussian(X):
    n = np.size(X, axis=1)
    hNDmax = np.nan
    if n >= 2:
        C = np.cov(X.T)
        detSigma = np.linalg.det(C)
        parenthesis = detSigma * (2 * math.pi * math.exp(1)) ** n
        hNDmax = .5 * math.log2(parenthesis)

    return hNDmax


# Η συνάρτηση αυτή κρύβει το κείμενο που τυπώνεται στην κονσόλα
# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

