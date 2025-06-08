# Repository to reproduce numerical experiments 

> BO_UCB_GRID_ERROR_BAR.py contains Python code to reproduce the results in Section 5 of the manuscript. 

> BO_UCB_GRID_UPDATED.png and BO_COMP_TIME_GRID_UPDATED contain the updated figures of Figures 1 and 2 in the submitted manuscript.

> BO_MICHALEWICZ_CUM_REG.png and BO_MICHALEWICZ_COMP_TIME.png demonstrate the cumulative regret and computation time comparison of GP-UCB algorithms based on the following acquisition function solvers: uniform random grid search(Uniform), Nelder-Mead(NM), Conjugate Gradient Descent (CG), and L-BFGS-B on a 10-dimensional Michalewicz objective function.

> BO_UCB_GRID_automl_CUM_REG.pdf and BO_UCB_GRID_automl_COMP_TIME.pdf showcase the effectiveness of uniform random grid search as the acquisition function solver for an 11-dimensional machine learning model hyperparameter tuning problem.

> BO_UCB_GRID_CUM_REG_SMALL_INITIAL.png and BO_UCB_GRID_COMP_TIME_SMALL_INITIAL.png are simulation results with a smaller number of initial design points following the heuristics ($n = 5d$, note previously it was $n = 10d$) as well as a smaller number of iterations (so that the total number of function evaluations is ($T = 20d$). 

> automl_uniform_grid_size_exp.pdf is the experiment where we test the effect of using linearly increasing grid size versus a fixed grid size(size of 100) in the 11-dimensional machine learning hyperparameter tuning problem. 
