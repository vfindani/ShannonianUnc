import numpy as np
import math as mth


# Στην υπορουτίνα αυτή προσομοιώνεται η διαδικασία της φυσικής επιλογής
def natural_selection(POP, obj_f):
    population_size = np.size(POP, 0)
    # S είναι το άθροισμα των τιμών του κριτηρίου για κάθε άτομο
    s = np.sum(obj_f)
    # Αρχικοποείται ο πίνακας POP_temp όπου αποθηκεύεται το αποτέλεσμα της φυσικής επιλογής
    # Στην αρχή, τα στοιχεία αυτού του πίνακα είναι όλα μηδενικά
    POP_temp = np.zeros((population_size, np.size(POP, 1)), dtype=int)

    # Στο διάνυσμα P_survival αποθηκεύεται η πιθανότητα επιβίωσης του ατόμου i
    # Στο διάνυσμα CP_survival αποθηκεύεται η συσσωρετυική πιθανότητα επιβίωσης του ατόμου i
    P_survival = np.divide(obj_f, s)
    CP_survival = np.zeros(population_size, dtype=float)
    # Το πρώτο στοιχείο αυτών των δύο πινάκων είναι το ίδιο
    CP_survival[0] = P_survival[0]

    # Για κάθε άτομο (εκτός του 0-οστού) υπολογίζεται η πιθανότητα και η συσσωρευτική πιθανότητα
    for i in range(1, population_size):
        CP_survival[i] = CP_survival[i-1]+P_survival[i]

    # Παράγονται τυχαίοι αριθμοί πλήθους population_size στο διάστημα [0,1]
    # βάσει της ομοιόμορφης κατανομής. Οι αριθμοί αυτοί αποθηκεύονται στον πίνακα ball
    ball = np.random.uniform(0.0, 1.0, population_size)

    # np.savetxt("report_of_ball.txt", ball, "%10.5f")
    # np.savetxt("report_of_P_survival.txt", P_survival, "%10.5f")
    # np.savetxt("report_of_CP_survival.txt", CP_survival, "%10.5f")

    # Από εδώ γυρίζει η ρουλέτα!!
    for i in range(0, population_size):  # Η i-οστή ρίψη της μπάλας
        for j in range(0, population_size):  # Ελέγχεται αν η i-οστή ρίψη κατέληξε στο j-οστό άτομο
            if j == 0:  # Αν j=0, ελέγχεται το 0-οστό (1ο) άτομο.
                if CP_survival[0] >= ball[i]:  # Αν ισχύει η ανισότητα, το άτομο αυτό επιλέγεται
                    POP_temp[i, :] = POP[0, :]
            else:       # Ο έλεγχος συνεχίζει για τα υπόλοιπα άτομα (j>0).
                if CP_survival[j-1] <= ball[i] <= CP_survival[j]:  # Αν ισχύει η ανισότητα, το j-οστό άτομο επιλέγεται
                    POP_temp[i, :] = POP[j, :]

    return POP_temp


# Στην υπορουτίνα αυτή προσομοιώνεται η διαδικασία της διασταύρωσης
def crossover(POP, z):

    population_size = np.size(POP, 0)

    # Αν ο z είναι περιττός αριθμός (δηλαδή το υπόλοιπο του κατά την διαίρεση του με το 2 είναι ίσο με 1),
    # τότε αυξάνεται κατά 1 ώστε να γίνει άρτιος.
    if (z % 2) == 1:
        z = z+1

    # Το t είναι το μισό του z, δηλαδή είναι ο αριθμός των ζευγαριών των ατόμων που σχηματίζονται
    t = int(z/2)

    # Δημιουργείται ο πίνακας "προξενήτρα" (matchmaker) όπου δημιουργούνται τυχαία ζευγάρια ατόμων
    # Ο πίνακας αυτός έχει διαστάσεις t x 2 και σε κάθε σειρά έχει ένα ζευγάρι ατόμων
    # Τα στοιχεία του πίνακα βρίσκονται εντός του διαστήματος [0, population_size).
    # Για να δημιουργηθεί ο πίνακας matchmaker ακολουθείται η εξής διαδικασία:
    matchmaker = np.arange(population_size, dtype=int)  # Δημιουργείται ένα διάνυσμα με στοιχεία [0, 1, ... pop_size-1]
    np.random.shuffle(matchmaker)  # Τα στοιχεία του διανύσματος αυτού ανακατεύονται τυχαία
    # Τα πρώτα population_size-z στοιχεία του διανύσματος matchmaker θα διαγραφούν ώστε αυτό να αποκτήσει μήκος ίσο με z
    matchmaker = np.delete(matchmaker, slice(population_size-z))
    # Το διάνυσμα matchmaker μετατρέπεται σε πίνακα t επί 2
    matchmaker.resize((t, 2))
    # print(matchmaker)

    # Δημιουργείται ο πίνακας cutting_point όπου ορίζονται τα σημεία κοπής των χρωμοσωμάτων
    # Τα στοιχεία του πίνακα cutting_point είναι ο αριθμός των γονιδίων βρίσκονται στο διάστημα [1, np.size(POP, 1))
    # Αν cutting_point=1, το αντίστοιχο ζεύγος χρωμ. κόβεται μεταξύ των γονιδίων 0 και 1
    # Αν cutting_point=x, το αντίστοιχο ζεύγος χρωμ. κόβεται μεταξύ των γονιδίων x-1 και x
    cutting_point = np.random.randint(1, np.size(POP, 1), size=t)
    # print(cutting_point)

    for i in range(0, t):  # Για το i-οστό ζευγάρι χρωμοσωμάτων
        rna = np.zeros(cutting_point[i])  # Αρχικοποιείται ο πίνακας rna όπου περιέχεται το τμήμα του χρωμοσώματος
        # που μεταφέρεται από το 1ο χρωμόσωμα στο 2o
        for j in range(0, cutting_point[i]):  # Για κάθε γονίδιο μέχρι το cutting point του i-οστού ζεύγους
            rna[j] = POP[matchmaker[i, 0], j]
            POP[matchmaker[i, 0], j] = POP[matchmaker[i, 1], j]
            POP[matchmaker[i, 1], j] = rna[j]

    return POP


# Στην υπορουτίνα αυτή προσομοιώνεται η διαδικασία της μετάλλαξης
def mutation(POP, mutation_rate):

    # Δημιουργείται ένας πίνακας ίσων διαστάσεων με τον POP όπου τα στοιχεία του αντιπροσωπεύουν
    # την πιθανότητα κάποιο γονίδιο να μεταλλαχθεί. Αν r[i, j] < mutation_rate τότε το γονίδιο μεταλλάσεται
    # και από 1 γίνεται 0 ή το αντίστροφο.
    r = np.random.uniform(0.0, 1.0, POP.shape)
    # np.savetxt("report_of_r.txt", r, "%10.6f")
    for i in range(0, np.size(POP, 0)):
        for j in range(0, np.size(POP, 1)):
            if r[i, j] < mutation_rate:
                if POP[i, j] == 1:
                    POP[i, j] = 0
                elif POP[i, j] == 0:
                    POP[i, j] = 1

    return POP


# Σε αυτή την υπορουτίνα βρίσκονται τα καλύτερα number_of_elites άτομα του τρέχοντος πληθυσμού POP
# και αποθηκεύονται στον πίνακα elite. Επίσης ο πίνακας POP μειώνεται κατά number_of_elites άτομα
# αφού αφαιρουνται από αυτόν number_of_elites άτομα τυχαία
def find_elite_v2(POP, number_of_elites, obj_functions):
    population_size = np.size(POP, 0)
    genes = np.size(POP, 1)
    temp = np.zeros((population_size, genes+1), dtype=float)
    POP_trimmed = np.zeros((population_size-number_of_elites, genes), dtype=int)

    # Οι πίνακες POP και Obj_functions συγχωνεύονται στον πίνακα Temp...
    for i in range(0, population_size):
        temp[i, :genes] = POP[i, :]
        temp[i, genes] = obj_functions[i]
    #   print(elite_temp)

    # ... ο οποίος έπειτα ταξινομείται ώστε τα στοιχεία στην τελευταία σειρά του να έχουν αύξουσα σειρά
    # τα στοιχεία της τελευταίας σειράς είναι η συνάρτηση στόχου
    temp = temp[temp[:, genes].argsort()]
    # print(elite_temp)
    # np.savetxt("elite_temp.txt", elite_temp, "%10.5f")
    elite = temp[population_size-number_of_elites:population_size, 0:genes]
    elite = elite.astype(int)

    selected = np.arange(population_size, dtype=int)  # Δημιουργείται ένα διάνυσμα με στοιχεία [0, 1, ... pop_size-1]
    np.random.shuffle(selected)  # Τα στοιχεία του διανύσματος αυτού ανακατεύονται τυχαία
    # Τα πρώτα number_of_elites στοιχεία του διανύσματος selected
    # θα διαγραφούν ώστε αυτό να αποκτήσει μήκος ίσο με population_size-number_of_elites
    selected = np.delete(selected, slice(number_of_elites))
    # Έπειτα σχηματίζεται ο νέος πίνακας POP από τον οποίον έχουν αφαιρεθεί τυχάια number_of_elites άτομα
    for i in range(0, np.size(selected)):
        POP_trimmed[i, :] = temp[selected[i], 0:genes]

    return elite, POP_trimmed


def find_elite(POP, number_of_elites, obj_functions):
    population_size = np.size(POP, 0)
    genes = np.size(POP, 1)
    temp = np.zeros((population_size, genes+1), dtype=float)
    for i in range(0, population_size):
        temp[i, :genes] = POP[i, :]
        temp[i, genes] = obj_functions[i]

    # print(temp)
    temp = temp[temp[:, genes].argsort()]

    # np.savetxt("elite_temp.txt", elite_temp, "%10.5f")
    elite = temp[population_size-number_of_elites:population_size, 0:genes]
    elite = elite.astype(int)

    return elite


# Σε αυτή την υπορουτίνα τα καλύτερα άτομα του πίνακα elite
# αντικαθιστούν τυχαία number_of_elites σε αριθμό άτομα του πίνακα POP
def implement_elitism(POP, number_of_elites, elite):
    population_size = np.size(POP, 0)

    removed = np.arange(population_size, dtype=int)  # Δημιουργείται ένα διάνυσμα με στοιχεία [0, 1, ... pop_size-1]
    np.random.shuffle(removed)  # Τα στοιχεία του διανύσματος αυτού ανακατεύονται τυχαία
    # Τα πρώτα population_size-number_of_elites στοιχεία του διανύσματος removed
    # θα διαγραφούν ώστε αυτό να αποκτήσει μήκος ίσο με number_of_elites
    removed = np.delete(removed, slice(population_size - number_of_elites))
    # print(removed)

    for i in range(0, number_of_elites):
        POP[removed[i], :] = elite[i, :]

    return POP


# Η ακόλουθη ρουτίνα βρίσκει το μέτωπο Pareto του τρέχοντος πληθυσμού
# και το αντιγράφει στον πίνακα Pareto
# Με την υπορουτίνα αυτή υλοποιείται το χαρακτηριστικό του ελιτισμού στον πολυκριτηριακό ΓΑ
def find_pareto_population(POP, rank, obj_f, sigma2):

    population_size = np.size(POP, 0)
    genes = np.size(POP, 1)
    crit_number = np.size(obj_f, 1)

    # Φτιάχνω την πρώτη γραμμή του πίνακα paretos όπου αποθηκεύονται τα γονίδια των ατόμων που βρίσκονται στο
    # μέτωπο Pareto. Αυτό το κάνω απλώς για να δημιουργήσω τον πίνακα...
    paretos = np.zeros((1, genes), dtype=int)
    obj_of_paretos = np.zeros((1, crit_number), dtype=float)

    # Τα σημεία με μηδενική τάξη (rank) ανήκουν στο μέτωπο pareto
    for i in range(0, population_size):
        if rank[i] == 0:
            paretos = np.vstack((paretos, POP[i, :]))
            obj_of_paretos = np.vstack((obj_of_paretos, obj_f[i, :]))

    # .. και εντάσσω την 1η σειρά ως προς διαγραφή
    remove = np.zeros(1, dtype=int)
    # obj_of_paretos = np.delete(obj_of_paretos, 0, axis=0)

    pareto_size = np.size(paretos, 0)
    dist2 = np.zeros((pareto_size, pareto_size), dtype=float)

# Αγνοώ την πρώτη σειρά του πίνακα Paretos ξεκινώντας το i από το 1
    for i in range(1, pareto_size):
        for j in range(i+1, pareto_size):
            for k in range(0, crit_number):
                dist2[i, j] += (obj_of_paretos[i, k]-obj_of_paretos[j, k])**2
            if dist2[i, j] < sigma2:
                remove = np.concatenate((remove, [i]), axis=0)

    remove = np.unique(remove)

    print(np.size(remove))
    # διαγράφω τις σειρές του πίνακα paretos με δείκτες που δηλώνονται στον πίνακα remove
    paretos = np.delete(paretos, remove, axis=0)
    obj_of_paretos = np.delete(obj_of_paretos, remove, axis=0)
    # ξαναυπολογίζω τον αριθμό των ατόμων που ανήκουν στο μέτωπο Pareto
    pareto_size = np.size(paretos, 0)

    return paretos, pareto_size, obj_of_paretos
