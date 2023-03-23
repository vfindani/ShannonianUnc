import numpy as np
# from pymoo.model.problem import FunctionalProblem
# from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import Hydrological_Models
from timeit import default_timer as timer
from pymoo.visualization.scatter import Scatter
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.factory import get_algorithm
from pymoo.factory import get_problem, get_reference_directions
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination


def comp_algo(limits, model, criterion, dates, path):

    start = timer()
    theta_number = np.size(limits, 0)
    model_family, K = Hydrological_Models.unify_data(model, dates, path)
    lower_lim = limits[:, 0]
    upper_lim = limits[:, 1]
    # algorithm = get_algorithm(name_of_algo)
    algorithm = NSGA2(pop_size=200)

    # Παρακάτω ορίζονται οι παράμετροι για την μονοκριτηριακή βαθμονόμηση με απλό GA
    if np.size(criterion, axis=0) == 2:
        objs = [
        lambda x: 10.0-np.dot(getattr(Hydrological_Models, model_family)(np.array([x,]), 1, model, criterion, K),
                              (criterion[1, :]).astype(float))]

        # https://pymoo.org/algorithms/genetic_algorithm.html
        termination = SingleObjectiveDefaultTermination(
            x_tol=1e-4,
            cv_tol=1e-6,
            f_tol=1e-4,
            nth_gen=5,
            n_last=20,
            n_max_gen=1000,
            n_max_evals=100000
        )

    # Παρακάτω ορίζονται οι παράμετροι για την πολυκριτηριακή βαθμονόμηση με τον αλγόριθμο NSGA2
    elif np.size(criterion, axis=0) == 1:

        objs = [
        lambda x: 10-(getattr(Hydrological_Models, model_family)(np.array([x,]), 1, model, criterion, K))[0]]

        termination = MultiObjectiveSpaceToleranceTermination(tol=0.0025,
                                                              n_last=20,
                                                              nth_gen=20,
                                                              n_max_gen=None,
                                                              n_max_evals=None)
        # https://pymoo.org/algorithms/nsga2.html
        #algorithm = NSGA2(pop_size=100,
        #                  sampling=get_sampling("real_random"),
        #                  crossover=get_crossover("real_sbx"),
        #                  mutation=get_mutation("real_pm"),
        #                  eliminate_duplicates=True)

    problem = FunctionalProblem(theta_number, objs, constr_ieq=[], xl=lower_lim, xu=upper_lim)
    res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)
    end = timer()

    print("Best solution found: \nX = %s\nF = %s" % (res.X, 10.0-res.F))
    print("clock time=", end - start)
    np.savetxt("pareto_front_alternative.txt", (10.0-res.F), "%10.4f")

    if np.size(criterion, axis=0) == 1:
        plot = Scatter()
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plot.add(res.F, color="red")
        plot.show()
