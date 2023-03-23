import Manipulate_Population
import Evolutionary_Operators
import numpy as np
import Hydrological_Models
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def Genetic_Algorithm(limits, model, criterion, dates, basin_info, population_size, convergence):
    # +++++++++++ Ορισμός των παράμετρων του ΓΑ ++++++++++++++++++++
    # https://www.youtube.com/watch?v=uzPT9dGgeTs&ab_channel=Steffix
    generations = 500
    max_non_improved = 20
    cross_rate = 0.85  # Δεκαδική τιμή του ποσοστού
    crossed = int(cross_rate*population_size)
    mutation_rate = 0.025  # Δεκαδική τιμή του ποσοστού
    elite_rate = 0.05  # Δεκαδική τιμή του ποσοστού
    number_of_elites = int(elite_rate * population_size)

    print("Simple Genetic Algorithm Initiation...")
    start = timer()

    # +++++++++++ Ορισμός των χαρακτηριστικών του μοντέλου προς βαθμονόμηση ++++++
    # Στον πίνακα limits αποθηκεύονται τα άνω και κάτω όρια των παραμέτρων
    theta_number = np.size(limits, 0)
    # theta_number είναι ο αριθμός των παραμέτρων του προβλήματος

    # +++++++++++++++++++ ΑΡΧΙΚΟΠΟΙΗΣΗ +++++++++++++++++++
    # Στον πίνακα POP αποθηκεύονται τα γονίδια του κάθε ατόμου του πληθυσμού άρα έχει διαστάσεις
    # population_size x τον αριθμό των γονιδίων ανά άτομο. Στην αρχή υποθέτουμε πως τα γονίδια ανά άτομο
    # είναι 2. Μέσω της υπορουτίνας Initialize_Population θα υπολογισθεί ο αριθμός των γονιδίων ανά άτομο.
    POP = np.ones((population_size, 2), dtype=int)
    # Στον πίνακα ms αποθηκεύεται ο αριθμός των γονιδίων ms[i] που χρειάζεται για να κωδικοποιηθεί η παράμετρος i
    ms = np.zeros(theta_number, dtype=int)
    # Στον πίνακα cms αποθηκεύονται τα σημεία στο χρωμόσωμα όπου αλλάζει η ομάδα των γονιδίων που κωδικοποιεί
    cms = np.zeros(theta_number+1, dtype=int)  # διαφορετική παράμετρο.
    # Ορίζεται ο πίνακας phenotype όπου αποθηκεύεται η αριθμητική τιμή των γονιδίων κάθε ατόμου
    phenotype = np.zeros((population_size, theta_number), dtype=float)
    # Ορίζεται ο πίνακας history όπου αποθηκεύεται η καλύτερη λύση κάθε γενιάς
    history = np.zeros((generations, theta_number+1))
    # Δημιουργείται ο αρχικός πληθυσμός και αποθηκεύεται στον πίνακα POP
    POP, ms, cms = Manipulate_Population.initial_chromo_gen(limits, ms, cms, population_size, POP)

    # Δημιουργείται ο πίνακας όπου αποθηκεύεται η τιμή της ολικής στοχικής συνάρτησης για κάθε άτομο του πληθυσμού
    # obj_functions = np.zeros(population_size, dtype=float). Ο αριθμός των γενιών που έχουν περάσει από την προς στιγμήν καλύτερη λύση
    best_solution = history[0, theta_number]
    best_location = history[0, :theta_number]
    gen_of_best = 0

    model, X = Hydrological_Models.unify_data(model, dates, basin_info)

    partial_obj = np.zeros((population_size, np.size(criterion, axis=1)), dtype=float)

    # ++++++++++++++ ΞΕΚΙΝΑΝΕ ΟΙ ΕΠΑΝΑΛΗΨΕΙΣ +++++++++++++++++++++++++++++++

    for k in range(0, generations):
        # Αποκωδικοποιείται ο φαινότυπος του πληθυσμού της k γενιάς
        phenotype = Manipulate_Population.decode2(phenotype, POP, limits, ms, cms)
        # Ανανεώνεται η τιμή των διάφορων κριτηρίων για κάθε φαινότυπο του τρέχοντος πληθυσμού
        # Το ποια συνάρτηση θα κληθεί από το module Hydrological Models εξαρτάται από την μεταβλητή model_family
        partial_obj = getattr(Hydrological_Models, model)(phenotype, population_size, criterion, X, partial_obj)
        # Η ολική στοχική συνάρτηση είναι το εσωτερικό γινόμενο των μερικών στοχικών συναρτήσεων με τα αντίστοιχα βάρη.Όταν έχουμε ένα κριτήριο εξ ορισμού obj_functions=partial_obj
        obj_functions = np.dot(partial_obj, (criterion[1, :]).astype(float))
        # Εντοπίζεται η ελιτ της τρέχουσας γενιάς
        elite = Evolutionary_Operators.find_elite(POP, number_of_elites, obj_functions)
        # np.savetxt("elite.txt", elite, "%10.0f")
        # Ο τελεστής της φυσικής επιλογής εφαρμόζεται στον πίνακα POP βάσει του κριτηρίου αξιόλησης
        POP = Evolutionary_Operators.natural_selection(POP, obj_functions)
        # np.savetxt("report_of_natural_selection.txt", POP, "%10.0f")
        POP = Evolutionary_Operators.crossover(POP, crossed)
        # np.savetxt("report_of_crossover.txt", POP, "%10.0f")
        POP = Evolutionary_Operators.mutation(POP, mutation_rate)
        # np.savetxt("report_of_mutation.txt", POP, "%10.0f")
        POP = Evolutionary_Operators.implement_elitism(POP, number_of_elites, elite)
        # np.savetxt("report_of_elitism.txt", POP, "%10.0f")

    # Στο τέλος της γενιάς προστίθεται η γραμμή k στον πίνακα history
    # history[k, :] = Manipulate_Population.herodotus(POP, phenotype, obj_functions)
        max_index = np.argmax(obj_functions)  # Ο δείκτης στον πίνακα obj_ όπου συναντάμε το καλύτερο άτομο της γενίας k
        history[k, :] = np.concatenate((phenotype[max_index, :], obj_functions[max_index]), axis=None)

    # Αν βρεθεί μία λύση καλύτερη της τρέχουσας βέλτιστής τότε αυτή γίνεται βέλτιση.
    # Επίσης, αποθηκεύεται ο φαινότυπος της και η τιμή της
        if (history[k, theta_number] - best_solution) >= convergence:
            best_solution = history[k, theta_number]
            best_location = history[k, :theta_number]
            gen_of_best = k

        non_improved = k-gen_of_best
        print(k, (history[k, theta_number]), non_improved)

        # Έλεγχος σύγκλισης
        # Εάν η λύση δεν έχει βελτιωθεί μετά από generations/10 γενιές, ο κώδικας τερματίζει
        if non_improved == max_non_improved:
            break

    end = timer()

    # np.savetxt("report_of_history.txt", history, "%10.8f")
    print("")
    print("----------------------- CALIBRATION REPORT ------------------------")
    print("Criterion(s):", criterion[0, :])
    print("Dates = ", dates)
    print("Best objective function value = ", best_solution)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=5)
    print("Best vector = ", best_location)
    print("Best generation = ", gen_of_best)
    print("clock time = ", end-start)
    print("")

    return best_location, best_solution

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def Pareto_Genetic_Algorithm(limits, model, criterion, dates, path):
    # +++++++++++ Ορισμός των παράμετρων του ΓΑ από τον χρήστη ++++++++++++++++++++
    # population_size = int(input("Give the population size. "))
    population_size = 400
    # generations = int(input("Give the number of generations. "))
    generations = 200
    # cross_rate = float(input("Give the crossover rate [%]. "))
    cross_rate = 90
    crossed = int(cross_rate*population_size/100)
    # mutation_rate = float(input("Give the mutation rate [%]. "))
    # mutation_rate = mutation_rate/100
    mutation_rate = 0.01
    # elite_rate = float(input("Give the elite rate [%]. "))
    # elite_rate = 50
    # number_of_elites = int(elite_rate * population_size/100)
    # crossed = crossed-number_of_elites
    # sigma = float(input("Give the value of sigma. "))
    density = .01
    sigma = density/population_size
    sigma2 = sigma**2
    # Οι παρακάτω παράμετροι ελέγχουν την σύγκλιση του αλγορίθμου
    interval_of_checking = 20  # Ανά πόσες γενιές ελέγχω αν το μέτωπο έχει αυξηθεί σε πληθυσμό κατά 10%
    conv_limit = 0.001
    last_centroid = np.zeros(np.size(criterion, axis=1), dtype=float)
    pareto_number_of_last_checked = 0
    pdp = .5

    print("Pareto Genetic Algorithm Initiation...")
    start = timer()

    # +++++++++++ Ορισμός των χαρακτηριστικών του μοντέλου προς βαθμονόμηση ++++++
    # Στον πίνακα limits αποθηκεύονται τα άνω και κάτω όρια των παραμέτρων
    theta_number = np.size(limits, 0)

    # +++++++++++++++++++ ΑΡΧΙΚΟΠΟΙΗΣΗ +++++++++++++++++++
    # Στον πίνακα POP αποθηκεύονται τα γονίδια του κάθε ατόμου του πληθυσμού άρα έχει διαστάσεις
    # population_size x τον αριθμό των γονιδίων ανά άτομο. Στην αρχή υποθέτουμε πως τα γονίδια ανά άτομο
    # είναι 2. Μέσω της υπορουτίνας Initialize_Population θα υπολογισθεί ο αριθμός των γονιδίων ανά άτομο.
    POP = np.ones((population_size, 2), dtype=int)
    # Στον πίνακα ms αποθηκεύεται ο αριθμός των γονιδίων ms[i] που χρειάζεται για να κωδικοποιηθεί η παράμετρος i
    ms = np.zeros(theta_number, dtype=int)
    # Στον πίνακα cms αποθηκεύονται τα σημεία στο χρωμόσωμα όπου αλλάζει η ομάδα των γονιδίων που κωδικοποιεί
    cms = np.zeros(theta_number+1, dtype=int)  # διαφορετική παράμετρο.
    # Δημιουργείται ο αρχικός πληθυσμός και αποθηκεύεται στον πίνακα POP
    POP, ms, cms = Manipulate_Population.initial_chromo_gen(limits, ms, cms, population_size, POP)
    # Ορίζεται ο πίνακας phenotype όπου αποθηκεύεται η αριθμητική τιμή των γονιδίων κάθε ατόμου
    phenotype = np.zeros((population_size, theta_number), dtype=float)
    model, X = Hydrological_Models.unify_data(model, dates, path)
    # Το X περιέχει είτε τα δεδομένα βροχόπτωσης, απορροής, εξατμισοδιαπνοής και θερμοκρασίας για το χρονικό
    # διάστημα βαθμονόμησης αν το μοντέλο είναι γραμμένο σε python [είτε τα άκρα του χρονικού διαστήματος
    # βαθμονόμησης αν το μοντέλο είναι γραμμένο σε R (δηλαδή τότε X=dates)] $$$$ δεν ισχύει πια

    last_checked_gen = 0

    partial_obj = np.zeros((population_size, np.size(criterion, axis=1)), dtype=float)

    # ++++++++++++++ ΞΕΚΙΝΑΝΕ ΟΙ ΕΠΑΝΑΛΗΨΕΙΣ +++++++++++++++++++++++++++++++

    for k in range(0, generations):
        # np.savetxt("report_of_POP.txt", POP, "%10.0f")

        # current_pop_size = np.size(POP, 0)
        # print(current_pop_size)
        # phenotype = np.zeros((current_pop_size, theta_number), dtype=float)
        # Αποκωδικοποιείται ο φαινότυπος του πληθυσμού της k γενιάς
        phenotype = Manipulate_Population.decode2(phenotype, POP, limits, ms, cms)
        # Ανανεώνεται η τιμή των διάφορων κριτηρίων για κάθε φαινότυπο του τρέχοντος πληθυσμού
        # obj_functions = getattr(Hydrological_Models, model_family)(phenotype, current_pop_size, model, criterion, X)
        obj_functions = getattr(Hydrological_Models, model)(phenotype, population_size, criterion, X, partial_obj)
        # np.savetxt("report_of_objectives.txt", obj_functions, "%10.8f")
        obj_value, rank = Manipulate_Population.just_pareto_ranking(obj_functions)
        # POP = Manipulate_Population.restore_pool_size(POP, population_size, obj_value)
        # Εντοπίζονται τα άτομα που ανήκουν στο μέτωπο Pareto και αποθηκεύονται στον πίνακα pareto_indi
        pareto_indi, pareto_number, XY_front = Evolutionary_Operators.find_pareto_population(
             POP, rank, obj_functions, sigma2)
        # print(POP.shape)
        # Ο τελεστής της φυσικής επιλογής εφαρμόζεται στον πίνακα POP βάσει του κριτηρίου αξιολόγησης
        POP = Evolutionary_Operators.natural_selection(POP, obj_value)
        # np.savetxt("report_of_natural_selection.txt", POP, "%10.0f")
        POP = Evolutionary_Operators.crossover(POP, crossed)
        # np.savetxt("report_of_crossover.txt", POP, "%10.0f")
        POP = Evolutionary_Operators.mutation(POP, mutation_rate)
        # np.savetxt("report_of_mutation.txt", POP, "%10.0f")
        # POP = np.concatenate((POP, pareto_indi), axis=0)
        POP = Evolutionary_Operators.implement_elitism(POP, np.size(pareto_indi, axis=0), pareto_indi)
        # np.savetxt("report_of_elitism.txt", POP, "%10.0f")

        # plt.clf()
        # plt.scatter(obj_functions[:, 0], obj_functions[:, 1])
        # plt.scatter(XY_front[:, 0], XY_front[:, 1])
        # plt.autoscale
        # plt.xlim((5, 10))
        # plt.ylim((0, 3))
        # plt.pause(.1)

        print(k, pareto_number, np.amax(obj_functions, axis=0))

        # ********************** ΕΛΕΓΧΟΣ ΣΥΓΚΛΙΣΗΣ [ΑΡΧΗ] **********************
        if k == (last_checked_gen+interval_of_checking):  # Όταν έρθει η ώρα για έλεγχο της σύγκλισης ανά πχ 10 γενιές
            last_checked_gen = k  # Δηλώνεται πως τσεκαρίστηκε η γενιά k
            print("Checking convergence...")
            current_centroid = np.average(XY_front, axis=0)
            Dx = current_centroid - last_centroid
            movement = np.sqrt(np.sum(Dx**2))
            print("DX of centroid = ", Dx, "Magnitude = ", movement)
            # Αν το κέντρο βάρους του μετώπου Pareto δεν έχει μετακινηθεί σημαντικά και ο αριθμός των ατόμων που ανήκουν
            # στο μέτωπο δεν έχει αυξηθεί τουλάχιστον κατά 10% από τον τελευταίο έλεγχο, τότε δεν υπάρχει σύγκλιση.
            if movement < conv_limit and pdp*population_size <= pareto_number <= 1.1 * pareto_number_of_last_checked:
                print("Algorithm converged and terminated!")
                break
            else:
                last_centroid = current_centroid
                pareto_number_of_last_checked = pareto_number
        # ********************** ΕΛΕΓΧΟΣ ΣΥΓΚΛΙΣΗΣ [ΤΕΛΟΣ] **********************

    end = timer()
    print("clock time =", end - start)
    np.savetxt("phenotypes_n_objectives.txt", np.hstack((phenotype, obj_functions)), "%10.4f")
    np.savetxt("rank_n_obj_value.txt", (np.concatenate(([rank], [obj_value]), axis=0)).T, "%10.4f")

    # Σχεδιάζεται το μέτωπο Pareto
    plt.scatter(XY_front[1:, 0], XY_front[1:, 1])
    np.savetxt("pareto_front.txt", XY_front[1:, :], "%10.4f")
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.show()


