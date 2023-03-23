import numpy as np
import math as mth
import Hydrological_Models


def initial_chromo_gen(limits, ms, cms, population_size, POP):

    digit_n = 5
    number_of_parameters = np.size(limits, 0)

    # Σε αυτό το for-block υπολογίζεται ο αριθμός ms[i] των γονιδίων που
    # απαιτούνται να κωδικοποιήσουν την i-οστή παράμετρο
    for i in range(0, number_of_parameters):
        ms[i] = np.ceil(np.log2(1.0+(10.0 ** digit_n)*(limits[i, 1]-limits[i, 0])))

    # Δημιουργείται ο πίνακας cms ο οποίος είναι το συσσωρευτικό άθροισμα του πίνακα ms ενώ το 1ο στοιχείο του
    # είναι πάντα το 0.
    cms = np.cumsum(ms, dtype=int)
    cms = np.concatenate((0, cms), axis=None)

    # Ο πίνακας POP αποκτά διαστάσεις (population_size x τον αριθμό genes των γονιδίων ανά άτομο)
    # όπου ο αριθμός genes είναι ίσος με το άθροισμα των στοιχείων του πίνακα ms
    # (και επομένως ο αριθμός genes είναι το τελευταίο στοιχείο του πίνακα cms)
    genes = cms[number_of_parameters]
    np.resize(POP, (population_size, genes))

    # Τα στοιχεία του πίνακα POP γίνονται τυχαία 0 ή 1, δηλαδή ακέραιοι μέχρι το 2.
    POP = np.random.randint(2, size=(population_size, genes))

    # print(POP)
    # print(ms[0], ms[1])
    return POP, ms, cms


# Η υπορουτίνα αυτή επιστρέφει ένα διάνυσμα με το καλύτερο άτομο του τρέχοντος πληθυσμού
def herodotus(POP, phenotype, F):

    current_max = F[0]
    max_index = 0
    for i in range(1, np.size(POP, 0)):
        if F[i] > current_max:
            current_max = F[i]
            max_index = i

    # best_cromo = POP[max_index, :]
    best_of_population = np.concatenate((phenotype[max_index, :], F[max_index]), axis=None)
    # print(best_of_population)
    return best_of_population


# Η υπορουτίνα αυτή λήφθηκε από την σελίδα https://www.geeksforgeeks.org/binary-decimal-vice-versa-python/
# και μετατρέπει δυαδικούς αριθμούς σε δεκαδικούς
def binaryToDecimal(binary):
    # binary1 = binary
    decimal, i, n = 0, 0, 0
    while binary != 0:
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1
    return decimal


# Η υπορουτίνα αυτή μετατέπει τον πίνακα με τα χρωμοσώματα POP στον πίνακα με τους φαινότυπους
# δηλαδή το χρωμόσωμα κάθε ατόμου αναλύεται και προκύπτουν οι τιμές των παραμέτρων που εκφράζει το άτομο αυτό.
def decode2(phenotype, POP, limits, ms, cms):

    for j in range(0, np.size(limits, 0)):  # Για την j-οστή παράμετρο

        # Αν τα όρια μίας μεταβλητής είναι ίσα τότε η τιμή της παραμέτρου είναι φιξαρισμένη και επομένως
        # η τιμή της είναι ίση με το άνω (ή κάτω) όριο για κάθε άτομο του πληθυσμού
        if limits[j, 0] == limits[j, 1]:
            phenotype[:, j] = limits[j, 0]
        # Αλλιώς κάθε άτομο στον πληθυσμό εκφράζει διαφορετική τιμή της παραμέτρου αυτής. Έτσι πρέπει για κάθε άτομο
        # το τμήμα του χρωμοσώματος που αντιστοιχεί στην παράμετρο αυτή να μεταφραστεί από το δυαδικό στο δεκαδικό
        # σύστημα και στην συνέχεια με τον τύπο (2.2) (βλέπε εργασία Σάββα, σελ 35) να παραχθεί ο φαινότυπος
        else:
            for i in range(0, np.size(POP, 0)):  # Για το i-οστό άτομο στον πληθυσμό
                binary_part = ""

                for k in range(cms[j], cms[j+1]):
                    binary_part = binary_part + str(POP[i, k])
                    # Τα cms[j] και cms[j+1] είναι μεταβλητές που αντιστοιχούν στις οριακές στήλες του πίνακα POP που
                    # περιέχουν τα γονίδια μίας μόνο παραμέτρου. Δηλαδή τα γονίδια POP[i,cms[j]:cms[j+1]] του ατόμου i
                    # κωδικοποιούν αντιστοιχούν στην j-οστή παράμετρο. Η αλληλουχία των γονιδίων αυτών εκφράζεται
                    # στο δεκαδικό σύστημα και αποθηκεύεται στην τιμή c...
                c = binaryToDecimal(int(binary_part, base=10))
                # .... Έπειτα, μέσω του παρακάτω τύπου δημιουργείται ο φαινότυπος του ατόμου i για κάθε παράμετρο j
                phenotype[i, j] = limits[j, 0]+(c/(2**ms[j]-1))*(limits[j, 1]-limits[j, 0])
                # Το c πρώτα διαιρείται με το 2^ms-1 και μετά το αποτέλεσμα πολλαπλασιάζεται με το
                # limits[j, 1]-limits[j, 0]. Αυτό γίνεται ώστε να μην προκύψουν μεγάλοι ακέραιοι
                # και γίνουν λάθη overflow
    return phenotype


def pareto_ranking(obj_function, sigma2):

    population_size = np.size(obj_function, axis=0)
    crit_number = np.size(obj_function, axis=1)
    delta = np.zeros((population_size, population_size), dtype=int)
    dist2 = np.zeros((population_size, population_size), dtype=float)
    spreading = np.eye(population_size, dtype=float)

    # Η παρακάτω διαδικασία περιγράφεται στην διδακτορική διατριβή του Ευστρατιάδη σελ 135
    # με την διαφορά πως εδώ εφαρμόζεται για την ταυτόχρονη μεγιστοποίηση όλων των κριτηρίων ενώ στην
    # διδακτορική διατριβή εφαρμόζεται για την ελαχιστοποίηση τους
    for i in range(0, population_size):
        for j in range(i+1, population_size):
            for k in range(0, crit_number):
                if obj_function[i, k] > obj_function[j, k]:
                    delta[i, j] += 1
                    delta[j, i] += -1
                elif obj_function[i, k] < obj_function[j, k]:
                    delta[i, j] += -1
                    delta[j, i] += 1
                # Υπολογίζεται η απόσταση στο πεδίο τιμών μεταξύ των λύσεων i και j
                dist2[i, j] += (obj_function[i, k]-obj_function[j, k])**2

    # np.savetxt("report_of_delta.txt", delta, "%10.0f")
    miou = ((np.absolute(delta)+delta)/(2*crit_number)).astype(int)
    # np.savetxt("report_of_miou.txt", miou, "%10.0f")
    # Στον πίνακα power αποθηκεύεται η ισχύς του κάθε ατόμου η οποία ορίζεται
    # ως ο αριθμός των ατόμων επί των οποίων αυτό κυριαρχεί
    power = np.sum(miou, axis=1)
    # np.savetxt("report_of_power.txt", power, "%10.0f")
    # Η τάξη ενός ατόμου είναι το άθροισμα των τιμών ισχύος των ατόμων από τα οποία κυριαρχείται.
    rank = np.matmul(miou.T, power)
    # np.savetxt("rank.txt", rank, "%10.0f")

    # Το στοιχείο spreading[i,j] εκφράζει την απόσταση μεταξύ των σημείων i και j
    # Όταν spreading[i,j]=1 τότε τα σημεία i και j ταυτίζονται.
    # Όταν spreading[i,j]=0 τότε τα σημεία i και j απέχουν απόσταση μεταλύτερη από sigma
    for i in range(0, population_size):
        for j in range(i+1, population_size):
            if dist2[i, j] <= sigma2:
                if rank[i] <= rank[j]:
                    spreading[j, i] = 1.0 - (dist2[i, j] / sigma2) ** .5
                else:
                    spreading[i, j] = 1.0 - (dist2[i, j] / sigma2) ** .5

    aggr_spreading = np.sum(spreading, axis=1)
    # np.savetxt("obj_function.txt", obj_function, "%10.3f")
    # np.savetxt("dist2.txt", dist2, "%10.3f")
    # np.savetxt("spreading.txt", spreading, "%10.3f")
    # np.savetxt("aggr_spreading.txt", aggr_spreading, "%10.2f")

    # Για τα μη κυριαρχούμενα άτομα το value τείνει να μεγαλώσει
    # Δηλαδή για τα "καλά" άτομα χαμηλού rank το value είναι μεγάλο
    # value = 100.0/((1.0+rank)*(aggr_spreading**.5))
    # value = 100.0/(1.0+rank)-10*(aggr_spreading-1)/(population_size-1)
    value = 100.0 / (1.0 + rank)
    # np.savetxt("value.txt", value, "%10.2f")

    return value, rank


def just_pareto_ranking(obj_function):
    population_size = np.size(obj_function, axis=0)
    crit_number = np.size(obj_function, axis=1)
    delta = np.zeros((population_size, population_size), dtype=int)

    # Η παρακάτω διαδικασία περιγράφεται στην διδακτορική διατριβή του Ευστρατιάδη σελ 135
    # με την διαφορά πως εδώ εφαρμόζεται για την ταυτόχρονη μεγιστοποίηση όλων των κριτηρίων ενώ στην
    # διδακτορική διατριβή εφαρμόζεται για την ελαχιστοποίηση τους
    for i in range(0, population_size):
        for j in range(i + 1, population_size):
            for k in range(0, crit_number):
                if obj_function[i, k] > obj_function[j, k]:
                    delta[i, j] += 1
                    delta[j, i] += -1
                elif obj_function[i, k] < obj_function[j, k]:
                    delta[i, j] += -1
                    delta[j, i] += 1

    miou = ((np.absolute(delta) + delta) / (2 * crit_number)).astype(int)
    # Στον πίνακα power αποθηκεύεται η ισχύς του κάθε ατόμου η οποία ορίζεται
    # ως ο αριθμός των ατόμων επί των οποίων αυτό κυριαρχεί
    power = np.sum(miou, axis=1)
    # Η τάξη ενός ατόμου είναι το άθροισμα των τιμών ισχύος των ατόμων από τα οποία κυριαρχείται.
    rank = np.matmul(miou.T, power)

    # Για τα μη κυριαρχούμενα άτομα το value τείνει να μεγαλώσει
    # Δηλαδή για τα "καλά" άτομα χαμηλού rank το value είναι μεγάλο
    value = 100.0 / (1.0 + rank)

    return value, rank


# Σε αυτή την υπορουτίνα βρίσκονται τα καλύτερα desired_n άτομα του πίνακα POP βάσει της τιμή του obj_function
# και αποθηκεύονται στον πίνακα elite. Τα χειρότερα population_size-desired_pop άτομα αφαιρούνται από τον πίνακα
# POP ώστε σε αυτόν να έχουν αποθηκευτεί desired_pop άτομα.
def restore_pool_size(POP, desired_pop, obj_functions):
    population_size = np.size(POP, 0)
    genes = np.size(POP, 1)
    temp = np.zeros((population_size, genes + 1), dtype=float)
    # POP_trimmed = np.zeros(desired_pop, genes), dtype=int)

    # Οι πίνακες POP και Obj_functions συγχωνεύονται στον πίνακα Temp...
    for i in range(0, population_size):
        temp[i, :genes] = POP[i, :]
        temp[i, genes] = obj_functions[i]

    # ... ο οποίος έπειτα ταξινομείται ώστε τα στοιχεία στην τελευταία σειρά του να έχουν αύξουσα σειρά
    # τα στοιχεία της τελευταίας σειράς είναι η συνάρτηση στόχου
    temp = temp[temp[:, genes].argsort()]
    # print(elite_temp)
    # np.savetxt("elite_temp.txt", elite_temp, "%10.5f")

    # Οι τελευταίες desired_pop γραμμές του πίνακα temp θα συνεχίσουν στον επόμενο πληθυσμό
    POP_trimmed = temp[population_size-desired_pop:population_size, 0:genes]
    POP_trimmed = POP_trimmed.astype(int)

    return POP_trimmed


