import numpy as np
import Entropy
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems.functional import FunctionalProblem


def Main():
    path = 'data/Timeseries_of_stations_Pili.txt'
    data = np.genfromtxt(path, dtype=None, delimiter=',', encoding="UTF-8")
    maxmin = -1.0  # Η μεταβλητή αυτή έχει την τιμή -1 αν μεγιστοποιώ το AvInf και την τιμή 1 αν το ελαχιστοποιώ

    # Διαγράφω από τον πίνακα data την 1η γραμμή που περιλαμβάνει τα ονόματα των στηλών και κάθε γραμμή i με ελλειπή στοιχεία σε οποιαδήποτε στήλη j.
    data = data[1:, :]
    data[data == ''] = np.nan
    data = data.astype(float)
    data = data[~np.isnan(data[:, - 1])]  # Αφαιρώ τις γραμμές με στοιχεία nan στην τελευταία στήλη
    data[data <= 0] = 0.01
    data = np.log(data)  # Παίρνω τους λογαρίθμους όλων των τιμων γιατί οι συναρτήσεις κοινής πληροφορίας δουλεύουν καλύτερα για λογαριθμικές κατανομές

    timeseries = data[:, :-2]  # Μέχρι την τελευταία στήλη έχουμε παρατηρήσεις για κάθε σταθμό. Κάθε σταθμός είναι μία στήλη.
    # T = data[:, -2]  # Η προτελευταία στήλη είναι παρατηρήσεις απορροής στην έξοδο της λεκάνης ΔΕΝ ΧΡΗΣΙΜΟΠΟΕΙΤΑΙ ΚΑΠΟΥ
    Q = data[:, -1]  # Η τελευταία στήλη είναι παρατηρήσεις απορροής στην έξοδο της λεκάνης

    station_n = np.size(timeseries, axis=1)  # Ο αριθμός των σταθμών είναι ο αριθμός των στηλών του πίνακα timeseries
    w_initial = np.ones(station_n)/station_n
    M, L = Entropy.adjustSteps(np.dot(timeseries, w_initial), Q)

    # https://pymoo.org/algorithms/genetic_algorithm.html
    parameter_n = station_n - 1  # Το πλήθος των παραμέτρων του προβλήματος βελτιστοποίησης είναι ο αριθμός των σταθμών μείον 1, λόγω του περιορισμού Σw=1
    algorithm = GA(pop_size=100, eliminate_duplicates=True)
    # objs = [lambda w: maxmin * Optimize_AvInf(np.dot(timeseries[:, :-1], w)+np.dot(timeseries[:, -1], 1.0-np.sum(w)), T, Q, C=3, IXYmin=0)]
    objs = [lambda w: maxmin * Entropy.Mutual_Information_2D(np.dot(timeseries[:, :-1], w) + np.dot(timeseries[:, -1], 1.0 - np.sum(w)), Q, M, L, noise=3,
                                                             suppress_negatives=True, ICA=False)]
    # Περιορισμός που υπαγορεύει ότι Σw - 1 < 0 ==> Σw < 1, με Σw να είναι το άθροισμα όλων των βαρών του σταθμού εκτός του τελευταίου. Δηλαδή ο περιορισμός στο πρόβλημα
    # βελτιστοποίησης περιορίζει να καταλήξουμε σε λύση όπου η επίδραση του τελευταίου σταθμού είναι αρνητική.
    constrains = [lambda w: np.sum(w)-1.0]
    problem = FunctionalProblem(parameter_n, objs, constr_ieq=constrains, xl=np.zeros(parameter_n), xu=np.ones(parameter_n))
    termination = DefaultSingleObjectiveTermination(
        xtol=1e-3,
        cvtol=1e-3,
        ftol=1e-3,
        period=20,
        n_max_gen=500,
        n_max_evals=100000)
    res = minimize(problem, algorithm, termination, save_history=True, verbose=True)

    if res.F is not None:
        Integrated_AvInf = maxmin * res.F
        Best_Areal_Distribution = res.X
        # Υπολογισμός της επίδρασης του τελευταίου σταθμού και προσθήκη στον πίνακα
        Best_Areal_Distribution = np.append(Best_Areal_Distribution, 1.0 - np.sum(Best_Areal_Distribution))
        # Υπολογισμός της πληροφορίας του αρχικού σετ. Η ελάχιστη τιμή της πραγματικής πληροφορίας είναι η τιμή Integrated_AvInf γιατί πρέπει Real_AvInf>Integrated_AvInf
        Real_AvInf, sigma_Real_AvInf = Entropy.Mutual_Information_ND(timeseries, Q, M, L, rep=40, noise=3, IXYmin=Integrated_AvInf)

        print("Best_Areal_Distribution =",  Best_Areal_Distribution)
        print("Integrated_(Compressed)_AvInf = ", Integrated_AvInf[0], " [bits]")
        print("Real_(Uncompressed)_AvInf = ", Real_AvInf, " [bits]")
        print("Information drop due to areal integration of time-series =", Real_AvInf - Integrated_AvInf, " [bits]")
    else:
        print("no optimal solution was found.")


def Optimize_AvInf(Pobs, Tobs, Qobs, C, IXYmin):

    # Δημιουργούνται οι υστερημένες χρονοσειρές αν C>0
    X = np.vstack((Pobs, Tobs))
    X = X.T
    k = np.size(X, axis=0)
    X1 = X
    if C > 0:
        Xlagged = np.zeros((k, np.size(X, axis=1), C))  # 3Δ Πίνακας διαστάσεων k, np.size(X,axis=1) και C
        for j in range(0, C):
            for i in range(0, k-j-1):
                Xlagged[i+j+1, :, j] = X[i, :]
            X1 = np.hstack((X1, Xlagged[:, :, j]))

    # !!!!!!!!!!!!!!!!!!!! Υπολογισμός της διαθέσιμης πληροφορίας !!!!!!!!!!!!!!!!!!!!
    # Δεν χρησιμοποιώ τις πρώτες C σειρές των πινάκων X1 και Qobs γιατί περιέχουν μηδενικά στοιχεία λόγω της μετατόπισης των υστερημένων χρονοσειρών
    AvInf, SigmaAvInf = Entropy.Mutual_Information_ND(X1[C:, :], Qobs[C:], M=int(k/4), L=int(k/16), rep=40, noise=3, IXYmin=IXYmin)

    return AvInf


if __name__ == "__main__":
    Main()
