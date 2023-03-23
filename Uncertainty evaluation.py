import numpy as np
import Entropy


def run_all_monthly_lumped_models(basin_info, criterion, cdates, vdates, PETcalc):
    limits_UTHBAL = np.array([[0.01, 100], [0, 1], [0, 1], [0, 1], [0, 1], [0, 10]])
    limits_GR2M = np.array([[1, 1500], [-10, 5], [0, 10]])
    limits_WBM = np.array([[0.001, 10], [0.001, 3000], [0.001, 10]])
    limits_Abulohom = np.array([[0, .1], [0, .01], [0, .01], [1, 3.99], [1, 1.99], [0, 10]])
    limits_Giakoumakis = np.array([[0.01, 100], [0, 1], [0, 10]])
    limits_TOPMODEL = np.array([[0, 0.001], [-8, 8], [0, 0.2], [0, 0.1], [0, 2], [0, 40], [0, 4000.0], [100.0, 2500.0], [0, 0.2], [0.0, 5.0], [720, 720]])

    if PETcalc:
        Entropy.diagnose_model("UTHBAL_Thornthwaite", limits_UTHBAL, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("GR2M_Thornthwaite", limits_GR2M, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("WBM_Thornthwaite", limits_WBM, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("Abulohom_Thornthwaite", limits_Abulohom, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("Giakoumakis_Thornthwaite", limits_Giakoumakis, basin_info, criterion, cdates, vdates)
        # Entropy.diagnose_model("TopModel_Thornthwaite", limits_TOPMODEL, basin_info, criterion, cdates, vdates)
    else:
        Entropy.diagnose_model("UTHBAL", limits_UTHBAL, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("GR2M", limits_GR2M, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("WBM", limits_WBM, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("Abulohom", limits_Abulohom, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("Giakoumakis", limits_Giakoumakis, basin_info, criterion, cdates, vdates)
        # Entropy.diagnose_model("TopModel", limits_TOPMODEL, basin_info, criterion, cdates, vdates)


def run_all_daily_lumped_models(basin_info, criterion, cdates, vdates, PETcalc):
    limits_TUWmodel = np.array([[0.9, 1.5], [0, 5], [1, 3], [-3, 1], [-2, 2], [0, 1], [0, 600], [0, 20], [0, 2], [2, 30], [30, 250], [1, 100], [0, 8], [0, 30], [0, 50]])
    limits_GR4J = np.array([[1, 1500], [-10, 5], [1, 500], [0.5, 4]])
    limits_HBV = np.array([[1, 1.5], [-2.5,2.5], [-2.5, 2.5], [0.5, 5], [50, 700], [0.3,1], [1,6], [0.05,0.99], [0.01, 0.8], [0.001, 0.15], [0, 100], [0, 6], [1, 3]])
    if PETcalc:
        Entropy.diagnose_model("TUWmodel_Thornthwaite", limits_TUWmodel, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("GR4J_Thornthwaite", limits_GR4J, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("HBV_Thornthwaite", limits_HBV, basin_info, criterion, cdates, vdates)
    else:
        Entropy.diagnose_model("TUWmodel", limits_TUWmodel, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("GR4J", limits_GR4J, basin_info, criterion, cdates, vdates)
        Entropy.diagnose_model("HBV", limits_HBV, basin_info, criterion, cdates, vdates)


def Main():
    open('Results_Uncertainty_Analysis.txt', 'w').close()
    open('Computed_Hydrographs.txt', 'w').close()
    with open('Results_Uncertainty_Analysis.txt', 'a') as the_file:
        the_file.write("UTHBAL, GR2M, WBM, Abulohom, Giakoumakis, TopModel for Sykia\n")
        the_file.write("ReqInf, AvInf, ExpInf, MaxExpInf, SigmaAvInf, phi, psi, delta, theta, ccrit, vcrit\n")

    criterion = np.array([["NSE"], [1]])

    basin_info = ["data\Data_Pili_01.txt", "", 39.453066918585]  # [path , dem_path , latitude]

    # cdates = ["1960-10-01", "1965-09-01"]
    # vdates = ["1965-10-01", "1994-09-01"]
    # with open('Results_Uncertainty_Analysis.txt', 'a') as the_file: the_file.write(cdates[0] + " " + cdates[1] + " & " + vdates[0] + " " + vdates[1] + " \n")
    # run_all_monthly_lumped_models(basin_info, criterion, cdates, vdates, PETcalc=True)
    #
    # cdates = ["1960-10-01", "1970-09-01"]
    # vdates = ["1970-10-01", "1994-09-01"]
    # with open('Results_Uncertainty_Analysis.txt', 'a') as the_file: the_file.write(cdates[0] + " " + cdates[1] + " & " + vdates[0] + " " + vdates[1] + " \n")
    # run_all_monthly_lumped_models(basin_info, criterion, cdates, vdates, PETcalc=True)
    #
    # cdates = ["1960-10-01", "1975-09-01"]
    # vdates = ["1975-10-01", "1994-09-01"]
    # with open('Results_Uncertainty_Analysis.txt', 'a') as the_file: the_file.write(cdates[0] + " " + cdates[1] + " & " + vdates[0] + " " + vdates[1] + " \n")
    # run_all_monthly_lumped_models(basin_info, criterion, cdates, vdates, PETcalc=True)
    #
    cdates = ["1960-10-01", "1982-09-01"]
    vdates = ["1982-10-01", "1994-09-01"]
    with open('Results_Uncertainty_Analysis.txt', 'a') as the_file: the_file.write(cdates[0] + " " + cdates[1] + " & " + vdates[0] + " " + vdates[1] + " \n")
    run_all_monthly_lumped_models(basin_info, criterion, cdates, vdates, PETcalc=True)
    #
    # cdates = ["1960-10-01", "1985-09-01"]
    # vdates = ["1985-10-01", "1994-09-01"]
    # with open('Results_Uncertainty_Analysis.txt', 'a') as the_file: the_file.write(cdates[0] + " " + cdates[1] + " & " + vdates[0] + " " + vdates[1] + " \n")
    # run_all_monthly_lumped_models(basin_info, criterion, cdates, vdates, PETcalc=True)
    #
    # cdates = ["1960-10-01", "1990-09-01"]
    # vdates = ["1990-10-01", "1994-09-01"]
    # with open('Results_Uncertainty_Analysis.txt', 'a') as the_file: the_file.write(cdates[0] + " " + cdates[1] + " & " + vdates[0] + " " + vdates[1] + " \n")
    # run_all_monthly_lumped_models(basin_info, criterion, cdates, vdates, PETcalc=True)

    # with open('Results_Uncertainty_Analysis.txt', 'a') as the_file:
    #     the_file.write("UTHBAL, GR2M, WBM, Abulohom, Giakoumakis, TopModel for Sykia\n")
    #     the_file.write("ReqInf, AvInf, ExpInf, MaxExpInf, SigmaAvInf, phi, psi, delta, theta, ccrit, vcrit\n")
    # basin_info = ["data\Data_Sarakina_01.txt", "", 39.472828134889]  # [path , dem_path , latitude]
    # cdates = ["1960-10-01", "1980-09-01"]
    # vdates = ["1980-10-01", "1991-03-01"]
    # with open('Results_Uncertainty_Analysis.txt', 'a') as the_file: the_file.write(cdates[0] + " " + cdates[1] + " & " + vdates[0] + " " + vdates[1] + " \n")
    # run_all_monthly_lumped_models(basin_info, criterion, cdates, vdates, PETcalc=True)


def Canva():
    path = "data\Data_Pili_01.txt"
    data = np.genfromtxt(path, dtype=None, delimiter=',', names=True, encoding="UTF-8")
    data = np.array([data["Date"].astype(str), data["R"], data["T"], data["Q"]]).T
    X = data[:, 1].astype(float)
    Z = data[:, 2].astype(float)
    Y = data[:, -1].astype(float)

    X[X <= 0] = 0.001
    Y[Y <= 0] = 0.001
    Z[Z <= 0] = 0.001
    X = np.log(X)
    Y = np.log(Y)
    Z = np.log(Z)

    # k = np.size(X, axis=0)
    # X1 = X
    # C = 10
    # Xlagged = np.zeros((k, np.size(X, axis=1), C))  # 3Δ Πίνακας διαστάσεων k, np.size(X,axis=1) και C
    # for j in range(0, C):
    #     for i in range(0, k-j-1):
    #         Xlagged[i+j+1, :, j] = X[i, :]
    #     X1 = np.hstack((X1, Xlagged[:, :, j]))

    # AvInf, sigma = Entropy.Mutual_Information_ND(X1[C:, :], Y[C:], M=30, L=None, rep=10, noise=3, IXYmin=0.0)
    AvInf1 = Entropy.Mutual_Information_2D(X, Y, M=30, L=8, suppress_negatives=True, noise=3, ICA=False)
    AvInf2 = Entropy.Mutual_Information_2D(Z, Y, M=30, L=8, suppress_negatives=False, noise=3, ICA=False)
    AvInf3 = Entropy.Mutual_Information_2D(X, Z, M=30, L=8, suppress_negatives=False, noise=3, ICA=False)
    print(AvInf1)
    print(AvInf2)
    print(AvInf3)


if __name__ == "__main__":
    # Main()
    Canva()





