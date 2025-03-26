from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.single.knapsack import create_random_knapsack_problem
from pymoo.optimize import minimize
import random
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

# seed_value = random.randint(1,200)
problem = create_random_knapsack_problem(n_items=30)

algorithm = NSGA2(
    pop_size=200,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),   # default mutation rate is 1/n, we can set using prob=
    eliminate_duplicates=True
)

res = minimize(
    problem,
    algorithm,
    ('n_gen', 100),
    verbose=True,
    seed=21
)

best_solution = res.X.astype(int)
best_solution_value = res.F
constraint_violation = res.CV

print("Best solution found (binary vector):", best_solution)
print("Objective value:", -best_solution_value)
print("Constraint violation (should be <= 0):", constraint_violation)
