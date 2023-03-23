import Genetic_Algorithm
# import Alt_Algorithms_Calibration
import numpy as np
import Hydrological_Models
import Manipulate_Population

# Στην αρχή πρέπει να ορισθούν 4 πίνακες: οι πίνακες limits, criterion, model, dates

# criterion = np.array([["NSE", "ExpInfLog"]])
# dates = ["1960-10-01", "1982-09-01"]

# model = "GR4J"
# limits = np.array([[1, 1500], [-10, 5], [1, 500], [0.5, 4]])
# model = "Giakoumakis"
# model = "Giakoumakis_Thornthwaite"
# limits = np.array([[0, 100], [0, 1], [0, 10]])
# model = "UTHBAL_Thornthwaite"
# limits = np.array([[0, 100], [0, 1], [0, 1], [0, 1], [0, 1], [0, 10]])
# model = "WBM"
# model = "WBM_Thornthwaite"
# limits = np.array([[0, 10], [0, 3000], [0, 10]])
# model ="Abulohom"
# model = "Abulohom_Thornthwaite"
# limits = np.array([[0, .1], [0, .01], [0, .01], [1, 3.99], [1, 1.99], [0, 10]])
# model = "GR2M"
# limits = np.array([[1, 1500], [-10, 5], [0, 10]])
# basin_info = ["data/Data_Mouzaki_01.txt", "data/dem_mouzaki.tif", 39.375152324221]
# dates = ["1960-10-01", "1982-09-01"]
# model = "TopModel_Thornthwaite"
# limits = np.array([[0, 0.001], [-8, 3], [0, 0.2], [0, 0.1], [0, 2], [0, 40], [0, 4000.0], [100.0, 2500.0], [0, 0.2], [0.0, 5.0], [720, 720]])
# model = "TUWmodel"
# limits = np.array([[0.9, 1.5], [0, 5], [1, 3], [-3, 1], [-2, 2], [0, 1], [0, 600], [0, 20], [0, 2], [2, 30], [30, 250], [1, 100], [0, 8], [0, 30], [0, 50]])

model = "HBV"
limits = np.array([[1, 1.5], [-2.5, 2.5], [-2.5, 2.5], [0.5, 5], [50, 700], [0.3, 1], [1, 6], [0.05, 0.99], [0.01, 0.8], [0.001, 0.15], [0, 100], [0, 6], [1, 3]])
basin_info = ["data\Data_Yermasogia_03.txt", "", np.nan]   # [path , dem_path , latitude]
cdates = ["1986-10-01", "1992-09-30"]
vdates = ["1992-10-01", "1997-09-30"]

validation = True
criterion = np.array([["NSE"], [1]])
validation_criterion = criterion

# Αν ο πίνακας criterion έχει δύο γραμμές τότε, τα κριτήρια που εισάγει ο χρήστης, θα βελτιστοποιηθούν συγχωνεύοντας
# τα με βάρη σε μία στοχική συνάρτηση. Επομένως σε αυτή την περίπτωση η βελτιστοποίηση είναι μονοκριτηριακή.
if np.size(criterion, axis=0) == 2:
    best_vector, ccrit = Genetic_Algorithm.Genetic_Algorithm(limits, model, criterion, cdates, basin_info, population_size=500, convergence=0.0001)

    if validation:
        # ΑΠΟ ΤΗΝ ΒΑΘΜΟΝΟΜΗΣΗ ΠΡΟΚΥΠΤΕΙ ΤΟ ΒΕΛΤΙΣΤΟ ΔΙΑΝΥΣΜΑ ΠΑΡΑΜΕΤΡΩΝ
        # ΕΤΣΙ ΥΠΟΛΟΓΙΖΕΤΑΙ ΤΟ ΚΡΙΤΗΡΙΟ validation_criterion για τις ημερομηνίες επαλήθευσης
        final_param = np.resize(best_vector, (1, np.size(best_vector)))
        model, X = Hydrological_Models.unify_data(model, vdates, basin_info)
        validation_crit = getattr(Hydrological_Models, model)(final_param, 1, validation_criterion, X, np.array([[0]], dtype=float))

        # ΕΚΤΥΠΩΣΗ ΤΗΣ ΤΙΜΗΣ ΤΟΥ ΕΠΙΛΕΓΜΕΝΟΥ ΚΡΙΤΗΡΙΟΥ ΓΙΑ ΤΗΝ ΠΕΡΙΟΔΟ ΕΠΑΛΗΘΕΥΣΗΣ
        print("------------------------ VALIDATION -------------------------")
        print(validation_criterion[0, :], " for ", vdates, " =", validation_crit[0, 0]-9)
        print("")

# Αντίθετα, εάν δεν εισαχθούν βάρη, ανεξάρτητα από τον αριθμό των κριτηρίων, η βελτιστοποίηση θα γίνει κατά Pareto
# Αυτό συμβαίνει όταν ο πίνακας criterion έχει μία γραμμή
elif np.size(criterion, axis=0) == 1:
    Genetic_Algorithm.Pareto_Genetic_Algorithm(limits, model, criterion, cdates, basin_info)

# elif mine is False:
#     # ['ga', 'brkga', 'de', 'nelder-mead', 'pattern-search', 'cmaes', 'nsga2', 'rnsga2', 'nsga3', 'unsga3', 'rnsga3',
#     # 'moead', 'ctaea']
#     # Alt_Algorithms_Calibration.comp_algo(limits, model, criterion, dates, path)

