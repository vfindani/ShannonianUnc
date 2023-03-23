import numpy as np
import Criterion_Catalog
from pyper import *
from PIL import Image


def Snow(Cm, T, P):
    # Με την ρουτίνα αυτή η υετόπτωση χωρίζεται σε βροχόπτωση και σε χιονόπτωση
    # Στην συνέχεια η χιονόπτωση χωρίζεται σε

    # Οι πίνακες SST, M και S γεμίζουν μηδενικά
    steps = np.size(P)
    SST = np.zeros(steps, dtype=float)
    M = np.zeros(steps, dtype=float)
    S = np.zeros(steps, dtype=float)

    for i in range(0, steps):
        if -10 < T[i] < 12.22:
            S[i] = P[i]/(1.0+1.61*np.power(1.35, T[i]))
        elif T[i] < -10:
            S[i] = P[i]

    R = P-S
    M0 = Cm*T

    for i in range(1, steps):
        SST[i] = np.maximum(SST[i-1]+S[i]-M0[i], 0)

    for i in range(1, steps):
        M[i] = SST[i-1]+S[i]

    M = np.minimum(M, M0)

    P = R + M

    return P


def Abulohom(phenotypes, population_size, criterion, data, partial_obj):

    Ep = data[:, 0]
    P = data[:, 1]
    T = data[:, 2]
    observed = data[:, 3]

    steps = np.size(P)
    w = np.zeros(steps, dtype=float)
    Ea = np.zeros(steps, dtype=float)
    Pa = np.zeros(steps, dtype=float)
    Rd = np.zeros(steps, dtype=float)
    Rg = np.zeros(steps, dtype=float)
    Rc = np.zeros(steps, dtype=float)
    m = 0.1 * np.ones(steps+1, dtype=float)

    crit_number = np.size(criterion, axis=1)
    # partial_obj = np.zeros((population_size, crit_number), dtype=object)

    for i in range(0, population_size):
        a1 = phenotypes[i, 0]
        a2 = phenotypes[i, 1]
        a3 = phenotypes[i, 2]
        b1 = int(phenotypes[i, 3])/2  # 0 < phenotypes[i, 3] < 3.99
        b2 = 2**(int(phenotypes[i, 4])-2)  # 1 < phenotypes[i, 4] < 3.99
        Cm = phenotypes[i, 5]

        # Στις παρακάτω γραμμές τρέχει ο κώδικας του μοντέλου Abulohom
        # Τρέχει πρώτα η ρουτίνα του χιονιού
        P0 = Snow(Cm, T, P)
        for k in range(0, steps):
            w[k] = P0[k]+m[k]
            Ea[k] = min(Ep[k]*(1.0-a3**(w[k]/Ep[k])), w[k])
            Pa[k] = P0[k]-Ea[k]*(1.0-np.exp(-P0[k]/Ea[k]))
            Rd[k] = a1*Pa[k]*(m[k]**b1)
            Rg[k] = a2*(m[k]**b2)
            Rc[k] = Rd[k] + Rg[k]
            m[k+1] = max((m[k] + P0[k]-Ea[k]-Rc[k]), 0.1)

        for j in range(0, crit_number):
            partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, Rc)

    return partial_obj


def UTHBAL(phenotypes, population_size, criterion, data, partial_obj):

    Ep = data[:, 0]
    P = data[:, 1]
    T = data[:, 2]
    observed = data[:, 3]

    steps = np.size(P)
    Nsmoist = 0.1 * np.ones(steps+1, dtype=float)
    Smoist = np.zeros(steps, dtype=float)
    AET = np.zeros(steps, dtype=float)
    Asmoist = np.zeros(steps, dtype=float)
    SR = np.zeros(steps, dtype=float)
    D = np.zeros(steps+1, dtype=float)
    Nmoist = np.zeros(steps+1, dtype=float)
    MR = np.zeros(steps, dtype=float)
    Qc = np.zeros(steps, dtype=float)

    crit_number = np.size(criterion, axis=1)
    # partial_obj = np.zeros((population_size, crit_number), dtype=float)

    for i in range(0, population_size):
        CN = phenotypes[i, 0]
        Kapa = phenotypes[i, 1]
        AAET = phenotypes[i, 2]
        CONMAR = phenotypes[i, 3]
        CONGROUND = phenotypes[i, 4]
        Cm = phenotypes[i, 5]
        Smax = 25400.0/CN-254.0

        # Στις παρακάτω γραμμές τρέχει ο κώδικας του μοντέλου UTHBAL
        # Τρέχει πρώτα η ρουτίνα του χιονιού
        P0 = Snow(Cm, T, P)
        for k in range(0, steps):
            Smoist[k] = P0[k]+Nsmoist[k]
            AET[k] = min(Smoist[k], Ep[k]*(1.0-AAET**(Smoist[k]/Ep[k])))
            Asmoist[k] = Smoist[k]-AET[k]
            SR[k] = max(0, (1.0-Kapa)*(Asmoist[k]-Smax))
            D[k+1] = max(0, Kapa*(Asmoist[k]-Smax))
            Nmoist[k+1] = Asmoist[k]-SR[k]-D[k+1]
            MR[k] = CONMAR*(Nmoist[k]+Nmoist[k+1])
            Nsmoist[k+1] = max(Asmoist[k]-SR[k]-D[k+1]-MR[k], 0)
            Qc[k] = SR[k]+CONGROUND*D[k]+MR[k]

        for j in range(0, crit_number):
            partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, Qc)

    return partial_obj


def WBM(phenotypes, population_size, criterion, data, partial_obj):

    Ep = data[:, 0]
    P = data[:, 1]
    T = data[:, 2]
    observed = data[:, 3]

    steps = np.size(P)
    Ea = np.zeros(steps, dtype=float)
    ST = 150 * np.ones(steps + 1, dtype=float)
    Qc = np.zeros(steps, dtype=float)

    crit_number = np.size(criterion, axis=1)
    # partial_obj = np.zeros((population_size, crit_number), dtype=float)

    for i in range(0, population_size):
        c = phenotypes[i, 0]
        SC = phenotypes[i, 1]
        Cm = phenotypes[i, 2]

        # Στις παρακάτω γραμμές τρέχει ο κώδικας του μοντέλου WBM
        # Τρέχει πρώτα η ρουτίνα του χιονιού
        P0 = Snow(Cm, T, P)
        for k in range(0, steps):
            Ea[k] = c*Ep[k]*np.tanh(P0[k]/Ep[k])
            const = ST[k]+P0[k]-Ea[k]
            Qc[k] = const*np.tanh(const/SC)
            ST[k + 1] = ST[k] + P0[k] - Ea[k] - Qc[k]

        for j in range(0, crit_number):
            partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, Qc)

    return partial_obj


def Giakoumakis(phenotypes, population_size, criterion, data, partial_obj):

    Ep = data[:, 0]
    P = data[:, 1]
    T = data[:, 2]
    observed = data[:, 3]

    steps = np.size(P)
    S = np.zeros(steps, dtype=float)
    Sn = np.zeros(steps, dtype=float)
    Dn = np.zeros(steps, dtype=float)
    DEY = np.zeros(steps + 1, dtype=float)  # Διαθέσιμη εδαφική υγρασία
    Qc = np.zeros(steps, dtype=float)

    crit_number = np.size(criterion, axis=1)
    # partial_obj = np.zeros((population_size, crit_number), dtype=float)

    for i in range(0, population_size):
        CN = phenotypes[i, 0]
        Kapa = phenotypes[i, 1]
        Cm = phenotypes[i, 2]
        Smax = 25400.0/CN-254.0

        # Στις παρακάτω γραμμές τρέχει ο κώδικας του μοντέλου WBM
        # Τρέχει πρώτα η ρουτίνα του χιονιού
        P0 = Snow(Cm, T, P)

        for k in range(0, steps):
            S[k] = DEY[k]+P0[k]-Ep[k]
            Sn[k] = min(Smax, S[k])
            Dn[k] = max(Kapa*(S[k]-Smax), 0)
            if S[k] < 0:
                DEY[k + 1] = 0
            else:
                DEY[k + 1] = Sn[k]
            Qc[k] = max((1-Kapa)*(S[k]-Smax), 0)

        for j in range(0, crit_number):
            partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, Qc)

    return partial_obj


def GR2M(phenotypes, population_size, criterion, data, partial_obj):

    Ep = data[:, 0]
    P = data[:, 1]
    T = data[:, 2]
    observed = data[:, 3]

    steps = np.size(P)
    S1 = np.zeros(steps, dtype=float)
    S2 = np.zeros(steps, dtype=float)
    S = np.zeros(steps + 1, dtype=float)  # Διαθέσιμη εδαφική υγρασία
    phi = np.zeros(steps, dtype=float)
    psi = np.zeros(steps, dtype=float)
    P1 = np.zeros(steps, dtype=float)
    P2 = np.zeros(steps, dtype=float)
    P3 = np.zeros(steps, dtype=float)
    R = np.zeros(steps + 1, dtype=float)
    R1 = np.zeros(steps, dtype=float)
    R2 = np.zeros(steps, dtype=float)
    Qc = np.zeros(steps, dtype=float)

    crit_number = np.size(criterion, axis=1)

    for i in range(0, population_size):
        X1 = phenotypes[i, 0]
        X2 = phenotypes[i, 1]
        Cm = phenotypes[i, 2]

        # Στις παρακάτω γραμμές τρέχει ο κώδικας του μοντέλου GR2M
        # Τρέχει πρώτα η ρουτίνα του χιονιού
        P0 = Snow(Cm, T, P)

        for k in range(0, steps):
            phi[k] = np.tanh(P0[k]/X1)
            S1[k] = (S[k]+X1*phi[k])/(1.0 + phi[k]*S[k]/X1)

            P1[k] = P0[k] + S[k] - S1[k]

            psi[k] = np.tanh(Ep[k]/X1)
            S2[k] = S1[k]*(1.0-psi[k])/(1.0 + psi[k] * (1.0-S1[k]/X1))

            S[k+1] = S2[k]/((1.0 + (S2[k]/X1)**3)**(1/3))

            P2[k] = S2[k] - S[k+1]
            P3[k] = P1[k] + P2[k]

            R1[k] = R[k] + P3[k]
            R2[k] = X2 * R1[k]
            Qc[k] = (R2[k]**2)/(R2[k]+60.0)
            R[k+1] = R2[k]-Qc[k]

        for j in range(0, crit_number):
            partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, Qc)

    return partial_obj


def GR4J(phenotypes, population_size, criterion, data, partial_obj):

    # generate a R instance
    r = R()
    # r = R(RCMD="C:\\Program Files\\R\\R-4.2.2\\bin\\R")

    # disable numpy & pandas in R
    r.has_numpy = False
    r.has_pandas = False

    # run R codes
    r.run('library(airGR)')
    r.run('gauged_data <- read.csv("data/medium.txt", header=TRUE)')
    r.run('FUN_MOD = RunModel_GR4J')

    observed = data[:, 3]
    k = np.size(observed, axis=0)
    r['k'] = k
    r.run('Ind_Run <- seq(1,k)')
    # r['start_date'] = "1986-10-01"
    # r['end_date'] = "1990-09-30"
    # r.run('Ind_Run <- seq(which(format(InputsModel$DatesR, format = "%Y-%m-%d") == start_date), which(format(InputsModel$DatesR, format = "%Y-%m-%d") == end_date))')

    r.run('InputsModel <- CreateInputsModel(FUN_MOD, DatesR = as.POSIXlt(gauged_data$Date, tz = "", "%Y-%m-%d",), Precip = gauged_data$R, PotEvap = gauged_data$PET)')
    r.run('RunOptions <- CreateRunOptions(FUN_MOD, InputsModel = InputsModel, IndPeriod_Run = Ind_Run)')

    crit_number = np.size(criterion, axis=1)
    for i in range(0, population_size):
        r['Param'] = phenotypes[i, :]
        r.run('OutputsModel <- RunModel(InputsModel, RunOptions, Param, FUN_MOD)')  # Τρέχει το μοντέλο
        simulated = np.array(r['OutputsModel$Qsim'])
        for j in range(0, crit_number):
            partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, simulated)

    return partial_obj


def Thornthwaite(T, phi):

    steps = np.size(T)
    year_n = int(steps/12)
    T0 = np.resize(T, (year_n, 12))
    T_montly_mean = np.average(T0, axis=0)
    I = np.sum((T_montly_mean/5)**1.514)
    a = 0.000000675*(I**3)-0.0000771*(I**2)+0.01792*I+0.49239
    number_of_day = np.array([31, 30, 31, 31, 28.25, 31, 30, 31, 30, 31, 31, 30])
    representive_day = np.array([289, 318, 345, 18, 46, 75, 105, 135, 162, 199, 229, 259])

    d = -.409*np.cos(2*np.pi*representive_day/365+.16)
    omega = np.arccos(-np.tan(phi)*np.tan(d))
    N = (24/np.pi)*omega
    Ld = (N*number_of_day)/360
    PET = np.zeros(steps)

    for i in range(0, steps):
        if T[i] > 0:
            PET[i] = 16*Ld[(i % 12)]*(10*T[i]/I)**a

    return PET


def TopModel(phenotypes, population_size, criterion, data, partial_obj):

    # generate a R instance
    r = R()
    # r = R(RCMD="C:\\Program Files\\R\\R-4.2.1\\bin\\R")

    # disable numpy & pandas in R
    r.has_numpy = False
    r.has_pandas = False

    # run R codes
    r.run('library(topmodel)')
    # r.run('library(Hmisc)')

    r.run('resol <- 5')  # <-----------------
    r.run('nHRU <- 10')  # Αριθμός των αδρομερών υδρολογικών μονάδων <-----------------
    r.run('n <- 10')  # number of classes; a higher number will result in a smoother histogram <-----------------

    r.run('DEM <- read.table("data/DEM.txt")')
    r.run('DEM <- as.matrix(DEM)')

    # Βρίσκω τους δείκτες του πιξελ της εξόδου της λεκάνης
    r.run('out_pos = which(abs(DEM) == min(abs(DEM)), arr.ind = TRUE)')
    r.run('y_out = as.integer(mean(out_pos[, 1]))')
    r.run('x_out = as.integer(mean(out_pos[, 2]))')

    r.run('DEM[DEM == -9999] <- NA')
    r.run('topindex <- topidx(DEM, resolution=resol)$atb')
    r.run('topind <- make.classes(topindex, nHRU)')

    # the delay function is a bit more tricky because this requires cumulative
    # fractions, but you generate it as follows:
    r.run('delay <- resol*flowlength(DEM, c(x_out,y_out))')
    r.run('delay <- make.classes(delay, n)')
    r.run('delay <- delay[n:1, ]')
    r.run('delay[, 2] <- c(0, cumsum(delay[1:(n - 1), 2]))')

    # Διαβάζει το αρχείο medium.txt με τις μετρήσεις για την λεκάνη
    r.run('gauged_data <- read.csv("data/medium.txt", header=TRUE)')
    # r.run('gauged_data_of_interest <- gauged_data[gauged_data$Date >= start_date & gauged_data$Date <= end_date,]')
    r.run('ETp <- 0.001 * gauged_data$PET')  # Οι μονάδες πρέπει να εισαχθούν σε m
    r.run('rain <- 0.001 * gauged_data$R')
    observed = 0.001 * data[:, 3]
    crit_number = np.size(criterion, axis=1)

    for i in range(0, population_size):
        r['Param'] = phenotypes[i, :]
        r.run('Qsim <- topmodel(Param, topind, delay, rain, ETp)')  # Τρέχει το μοντέλο
        try:
            simulated = np.array(r['Qsim'])
        except Exception:
            print("error: TopModel did not return Qsim.")
            simulated = np.zeros(np.size(observed))
            pass

        for j in range(0, crit_number):
            # Επιλέγεται από το αρχείο Criterion_Catalog ποιο κριτήριο θα υπολογισθεί βάσει της τιμής που είναι
            # αποθηκευμένη στην μεταβλητή criterion[0, j]. Αυτό είναι το j-οστό κριτήριο που θα υπολογισθεί
            partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, simulated)

    return partial_obj


def TUWmodel(phenotypes, population_size, criterion, data, partial_obj):

    # generate a R instance
    r = R()

    # disable numpy & pandas in R
    r.has_numpy = False
    r.has_pandas = False

    # run R codes
    r.run('library(TUWmodel)')

    r.run('gauged_data <- read.csv("data/medium.txt", header=TRUE)')
    r.run('prec <- gauged_data$R')
    r.run('airt <- gauged_data$T')
    r.run('ep <- gauged_data$PET')
    observed = data[:, 3]

    crit_number = np.size(criterion, axis=1)

    for i in range(0, population_size):
        r['Param'] = phenotypes[i, :]
        r.run('Qsim <- TUWmodel(prec, airt, ep, area=1, Param)')  # Τρέχει το μοντέλο
        simulated = r['Qsim$q']
        simulated = np.array(simulated)
        simulated = simulated[0, :]

        for j in range(0, crit_number):
            partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, simulated)

    return partial_obj


def HBV(phenotypes, population_size, criterion, data, partial_obj):

    # generate a R instance
    r = R()

    # disable numpy & pandas in R
    r.has_numpy = False
    r.has_pandas = False

    # run R codes
    r.run('library(HBV.IANIGLA)')

    crit_number = np.size(criterion, axis=1)
    observed = data[:, 3]

    r('init_snow = 20')
    r('init_soil = 100')
    r('init_routing = c(0, 0, 0)')

    r.run('gauged_data <- read.csv("data/medium.txt", header=TRUE)')

    for i in range(0, population_size):
        if 1.0 > phenotypes[i, 7] > phenotypes[i, 8] > phenotypes[i, 9] and phenotypes[i, 10] > phenotypes[i, 11]:  # Πρέπει να ισχύει αυτός ο περιορισμός στις παραμέτρους
            r['param_snow'] = phenotypes[i, 0:4]
            r['param_soil'] = phenotypes[i, 4:7]
            r['param_routing'] = phenotypes[i, 7:12]
            r['param_tf'] = phenotypes[i, 12]
            r.run('snow_module <- SnowGlacier_HBV(model=1, inputData=as.matrix(gauged_data[, c("T", "R")]), initCond = c(init_snow, 2), param = param_snow)')
            r.run('soil_module <- Soil_HBV(model=1,inputData=cbind(snow_module[, "Total"], gauged_data$"PET"), initCond = c(init_soil, 1), param = param_soil)')
            r.run('routing_module <- Routing_HBV(model=1, lake=F, inputData=as.matrix(soil_module[, "Rech"]), initCond = init_routing, param = param_routing)')
            r.run('tf_module <- UH(model=1, Qg=routing_module[, "Qg"], param = param_tf)')
            r.run('Qsim <- round(tf_module, 2)')
            simulated = r['Qsim']
            simulated = np.array(simulated)
            for j in range(0, crit_number):
                partial_obj[i, j] = getattr(Criterion_Catalog, criterion[0, j])(observed, simulated)
        else:
            for j in range(0, crit_number):
                partial_obj[i, j] = -9999.9

    return partial_obj


# Σε αυτή την ρουτίνα, εντοπίζονται δύο τύποι δεδομένων (δεδομένα με εξατμισοδιαπνοή ή χωρίς αυτή). Ο δεύτερος τύπος
# δεδομένων μετατρέπεται στον πρώτο με το μοντέλο του Thornthwaite.
# Επίσης, αν το μοντέλο είναι το TopModel, διαμορφώνω ένα αρχείο txt το οποίο θα μπορεί να το διαβάζει η γλώσσα R
# και μετατρέπω το αρχείο tiff της λεκάνης σε txt για να τα διαβάσει επίσης η R.
def unify_data(model, dates, basin_info):

    path = basin_info[0]
    dem_path = basin_info[1]
    latitude = basin_info[2]
    open('data/medium.txt', 'w').close()
    open('data/DEM.txt', 'w').close()

    data = np.genfromtxt(path, dtype=None, delimiter=',', names=True, encoding="UTF-8")

    if "Thornthwaite" in model:
        data = np.array([data["Date"].astype(str), data["R"], data["T"], data["Q"]]).T
        data, list_of_dates = trim_data(data, dates)
        latitude = latitude * 0.017453293  # Το γεωγραφικό πλάτος πρέπει να μετατραπεί από deg σε rad
        PET = Thornthwaite(data[:, 1], latitude)
        data_with_PET = np.vstack((PET, data.T))
        X = data_with_PET.T
        model = model.rstrip("_Thornthwaite")  # Αφαιρώ την λέξη Thornthwaite από το model
    else:
        data = np.array([data["Date"].astype(str), data["PET"], data["R"], data["T"], data["Q"]]).T
        X, list_of_dates = trim_data(data, dates)
        print("Latitude with value ", latitude, " rad is not needed.")

    r_models = ["TopModel", "GR4J", "TUWmodel", "HBV"]
    # Για όσα μοντέλα είναι στην R, διαμορφώνω ένα αρχείο txt το οποίο θα το διαβάζει η γλώσσα R
    if model in r_models:
        with open('data/medium.txt', 'a') as the_file:
            the_file.write("Date,PET,R,T,Q\n")
            for i in range(0, np.size(X, axis=0)):
                the_file.write(list_of_dates[i] + "," + X[i, 0].astype(str) + "," + X[i, 1].astype(str) + "," + X[i, 2].astype(str) + "," + X[i, 3].astype(str) + "\n")

    DEMed_models = ["TopModel"]
    # Μετατρέπω το DEM από tiff σε txt για να το διαβάσει το TopModel
    if model in DEMed_models:
        raster = np.array(Image.open(dem_path, mode='r', formats=None))
        np.savetxt("data/DEM.txt", np.flip(raster, 1), "%10.0f")

    return model, X


# Επιλέγω τα δεδομένα εντός του συγκεκριμένου εύρους ημερομηνιών
def trim_data(data, dates):
    a = int(np.array(np.where(data[:, 0] == dates[0])))
    b = 1 + int(np.array(np.where(data[:, 0] == dates[1])))
    numeric_data = data[a:b, 1:].astype(float)
    list_of_dates = data[a:b, 0].astype(str)

    return numeric_data, list_of_dates
