#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import itertools
import pandas as pd

from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling

class ShipLoadingProblem(ElementwiseProblem):
    def __init__(self, dataset, num_bays, num_tiers, num_rows, weight_delta=0):
        num_containers = len(dataset)
        n_var = num_bays*num_tiers*num_rows
        n_ieq_constr = 1 # Non repeating containers
        n_ieq_constr += num_bays*(num_tiers-1)*num_rows # Ordering of weights
        n_ieq_constr += num_bays*(num_tiers-1)*num_rows # Ordering of destinations
        super().__init__(
            n_var=n_var, 
            n_obj=3, 
            n_ieq_constr=n_ieq_constr,
            xl=-1, 
            xu=num_containers-1, 
            vtype=int
        )

        self.num_containers = num_containers
        self.num_bays = num_bays
        self.num_tiers = num_tiers
        self.num_rows = num_rows
        self.weight_delta = weight_delta

        self.num_spaces = num_bays * num_tiers * num_rows

        self.weights = dataset["Weight"].to_numpy()
        self.destinations = dataset["Destination"].to_numpy()

    def _evaluate(self, x, out, *args, **kwargs):
        x_rs = np.reshape(x, (self.num_bays, self.num_tiers, self.num_rows))

        # Maximum loading
        F_loaded_containers = np.count_nonzero(x_rs != -1)

        # Good balance
        F_x_balance = 0
        F_z_balance = 0
        
        for i, j, k in itertools.product(range(self.num_bays), range(self.num_tiers), range(self.num_rows)):
            w = self.weights[x_rs[i,j,k]] if x_rs[i,j,k] != -1 else 0
            F_x_balance += (i - ((self.num_bays-1) / 2)) * w
            F_z_balance += (k - ((self.num_rows-1) / 2)) * w

        F_x_balance = np.abs(F_x_balance)
        F_z_balance = np.abs(F_z_balance)
        
        G = []

        # No repeating containers
        containers = x[np.where(x != -1)]
        constraint = len(containers) - len(np.unique(containers))
        # num_containers - num_unique_containers - (1 if -1 in x)
        G.append(constraint)
        
        # Weight ordering
        for j in range(self.num_tiers-1):
            for i, k in itertools.product(range(self.num_bays), range(self.num_rows)):
                w_above = self.weights[x_rs[i,j+1,k]] if x_rs[i,j+1,k] != -1 else -10*self.weight_delta
                w_below = self.weights[x_rs[i,j,k]] if x_rs[i,j,k] != -1 else -10*self.weight_delta
                constraint = w_above - w_below - self.weight_delta
                G.append(constraint)

        # Destination ordering
        for j in range(self.num_tiers-1):
            for i, k in itertools.product(range(self.num_bays), range(self.num_rows)):
                if x_rs[i,j+1,k] == -1:
                    # If above is empty, its ok
                    constraint = -1
                elif x_rs[i,j,k] == -1:
                    # If below is empty and above is full, constraint unsatisfied
                    constraint = 1
                else:
                    # Check order of destination
                    d_above = self.destinations[x_rs[i,j+1,k]] #if x_rs[i,j+1,k] != -1 else -1
                    d_below = self.destinations[x_rs[i,j,k]]# if x_rs[i,j,k] != -1 else -1
                    constraint = 10*(int(d_above) - int(d_below))
                G.append(constraint)
                
        out["F"] = [-F_loaded_containers, F_x_balance, F_z_balance]
        out["G"] = G

# Define the problem-specific parameters
dataset = pd.read_csv("./containers1.csv")
num_bays = 8
num_tiers = 3
num_rows = 4

population_size = 128

problem = ShipLoadingProblem(dataset, num_bays, num_tiers, num_rows)

# Define the genetic algorithm settings
algorithm = NSGA2(
    pop_size=population_size,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True
)


# In[13]:


# After creating the algorithm object
print("Algorithm Population Size:", algorithm.pop_size)

# Solve the problem
res = minimize(problem,
               algorithm,
               termination=('n_gen', 20000),
               seed=34,
               save_history=False,
               pop_size=population_size,
               verbose=True)

# Retrieve the optimal solutions
best_solutions = res.X
best_objectives = res.F
print('The best solutions are:\n',best_solutions, end='\n ')
print('The best objectives are:\n',best_objectives, end='\n ')
print('Constraint violation:\n',res.CV, end='\n ')

# Visualize the loading plan of the best solution

best_loading_plan = best_solutions[0]
#best_loading_plan = np.random.choice(best_loading_plan, size=(num_bays*num_tiers*num_rows), replace=False)
best_loading_plan = np.reshape(best_loading_plan, (num_bays, num_tiers, num_rows))
print('The best loading plan is:\n',best_loading_plan, end='\n ')


# In[14]:


print(res.__dir__())
print(res.G[0,7:])


# In[18]:


import seaborn as sns

# Extract the top tier data
top_tier = best_loading_plan[:, -1, :]
front_row = best_loading_plan[:, :, -1]

# Replace container indices with their weights
dataset = pd.read_csv("./containers1.csv")
weights = dataset["Weight"].to_numpy()
dest = dataset["Destination"].to_numpy()
top_tier_weights = []
for i in top_tier.flatten():
    if i == -1:
        top_tier_weights.append(-1)
    else:
        top_tier_weights.append(weights[i])
top_tier_weights = np.array(top_tier_weights)
top_tier_weights = top_tier_weights.reshape(num_rows, num_bays)
top_tier_weights = np.flipud(top_tier_weights)  # Vertically invert the data

front_row_dest = []
for i in front_row.flatten():
    if i == -1:
        front_row_dest.append(-1)
    else:
        front_row_dest.append(dest[i])
front_row_dest = np.array(front_row_dest)
front_row_dest = front_row_dest.reshape(num_bays, num_tiers).T
#front_row_dest = np.flipud(front_row_dest)  # Vertically invert the data

# Plot the heatmap using seaborn
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(top_tier_weights, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Weight'}, ax=ax)
plt.xlabel("Bay")
plt.ylabel("Row")
plt.title("Weight distribution of the containers viewed from the top")
ax.invert_yaxis()  # Invert the y-axis

# Display the plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(front_row_dest, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Destination'}, ax=ax)
plt.xlabel("Bay")
plt.ylabel("Tier")
plt.title("Destination distribution of the containers viewed from the side")
ax.invert_yaxis()

plt.show()


# In[ ]:




