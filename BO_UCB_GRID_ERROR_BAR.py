import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import qmc  # For quasi–Monte Carlo sampling
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")
np.random.seed(20252026)

# =============================================================================
# 1. Objective Functions
# -----------------------------------------------------------------------------
# All functions are transformed so that the maximum value is 0.

# 1a. Negative Rastrigin (3d)
def neg_rast(x, A=10):
    x = np.asarray(x)
    d = x.size
    rastrigin_val = A * d + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return -rastrigin_val

# 1b. Hartmann3 (3d)
def hartmann3_raw(x):
    x = np.asarray(x)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381, 5743, 8828]])
    outer = 0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        outer += alpha[i] * np.exp(-inner)
    return -outer

def hartmann3(x):
    return -hartmann3_raw(x) - 3.86278  # MAX is at zero

# 1c. Hartmann4 (4d)
def hartmann4_raw(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5],
                  [0.05, 10, 17, 0.1],
                  [3, 3.5, 1.7, 10],
                  [17, 8, 0.05, 10]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124],
                         [2329, 4135, 8307, 3736],
                         [2348, 1451, 3522, 2883],
                         [4047, 8828, 8732, 5743]])
    outer = 0.0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        outer += alpha[i] * np.exp(-inner)
    return -outer

def hartmann4(x):
    return -hartmann4_raw(x) - 3.729840584485584

# 1d. Hartmann6 (6d)
def hartmann6_raw(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    outer = 0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2)
        outer += alpha[i] * np.exp(-inner)
    return -outer

def hartmann6(x):
    return -hartmann6_raw(x) - 3.32237

# 1e. Branin (2d)
def branin(x):
    x = np.asarray(x)
    x1, x2 = x
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    f = a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s
    return -f + 0.397887

# 1f. Levy5 (5d)
def levy5(x):
    x = np.asarray(x)
    d = x.size
    w = 1 + (x - 1) / 4.0
    term1 = np.sin(np.pi * w[0])**2
    term_sum = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term_last = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    f_val = term1 + term_sum + term_last
    return -f_val

# =============================================================================
# 2. UCB Acquisition Function
# -----------------------------------------------------------------------------
def ucb(x, gp, kappa):
    mu, sigma = gp.predict(x, return_std=True)
    return mu + kappa * sigma

# =============================================================================
# 3. Inner Optimization Methods
# -----------------------------------------------------------------------------
def optimize_acquisition_grid(gp, kappa, bounds, iteration):
    d = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    base_size = 100
    grid_size = base_size * (iteration + 1)
    np.random.seed(iteration)
    grid_points = lower + (upper - lower) * np.random.rand(grid_size, d)
    acq_values = ucb(grid_points, gp, kappa)
    best_idx = np.argmax(acq_values)
    return grid_points[best_idx]

def optimize_acquisition_quasi_newton(gp, kappa, bounds, n_restarts=20, tol=0.01):
    d = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    def min_obj(x):
        return -ucb(np.array(x).reshape(1, -1), gp, kappa)
    best_x = None
    best_acq_value = -np.inf
    for i in range(n_restarts):
        x0 = lower + (upper - lower) * np.random.rand(d)
        res = minimize(min_obj, x0=x0, bounds=bounds, method="L-BFGS-B",
                       options={'gtol': tol, 'maxiter': d*200})
        if res.success and (-res.fun > best_acq_value):
            best_acq_value = -res.fun
            best_x = res.x
    return best_x

def optimize_acquisition_nelder(gp, kappa, bounds, n_restarts=20, tol=0.01):
    d = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    def min_obj(x):
        return -ucb(np.clip(x, lower, upper).reshape(1, -1), gp, kappa)
    best_x = None
    best_acq_value = -np.inf
    for i in range(n_restarts):
        x0 = lower + (upper - lower) * np.random.rand(d)
        res = minimize(min_obj, x0=x0, method="Nelder-Mead",
                       options={'xatol': tol, 'fatol': tol})
        if res.success:
            candidate = np.clip(res.x, lower, upper)
            current_acq = -min_obj(candidate)
            if current_acq > best_acq_value:
                best_acq_value = current_acq
                best_x = candidate
    return best_x

def optimize_acquisition_cg(gp, kappa, bounds, n_restarts=20, tol=0.01):
    d = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    def min_obj(x):
        return -ucb(np.clip(x, lower, upper).reshape(1, -1), gp, kappa)
    best_x = None
    best_acq_value = -np.inf
    for i in range(n_restarts):
        x0 = lower + (upper - lower) * np.random.rand(d)
        res = minimize(min_obj, x0=x0, method="CG",
                       options={'gtol': tol, 'maxiter': d*200})
        if res.success:
            candidate = np.clip(res.x, lower, upper)
            current_acq = -min_obj(candidate)
            if current_acq > best_acq_value:
                best_acq_value = current_acq
                best_x = candidate
    return best_x

# =============================================================================
# 4. Bayesian Optimization Loop (with instantaneous regret recording)
# -----------------------------------------------------------------------------
def bayesian_optimization(objective, bounds, n_init=10, n_iter=30,
                          inner_optimizer='grid', X_init=None, Y_init=None):
    d = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    if X_init is None or Y_init is None:
        X_sample = lower + (upper - lower) * np.random.rand(n_init, d)
        Y_sample = np.array([objective(x) for x in X_sample])
    else:
        X_sample = X_init.copy()
        Y_sample = Y_init.copy()
    kernel = Matern(nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)
    best_values = [np.max(Y_sample)]
    instant_regrets = []
    for i in range(n_iter):
        effective_kappa = np.sqrt(np.log(i + 2))
        gp.fit(X_sample, Y_sample)
        if inner_optimizer == 'grid':
            x_next = optimize_acquisition_grid(gp, effective_kappa, bounds, iteration=i)
        elif inner_optimizer == 'quasi':
            x_next = optimize_acquisition_quasi_newton(gp, effective_kappa, bounds, n_restarts=10, tol=0.01)
        elif inner_optimizer == 'nelder':
            x_next = optimize_acquisition_nelder(gp, effective_kappa, bounds, n_restarts=10, tol=0.01)
        elif inner_optimizer == 'CG':
            x_next = optimize_acquisition_cg(gp, effective_kappa, bounds, n_restarts=10, tol=0.01)
        else:
            raise ValueError("Unknown inner_optimizer type.")
        y_next = objective(x_next)
        instant_regrets.append(0 - y_next)
        X_sample = np.vstack((X_sample, x_next))
        Y_sample = np.append(Y_sample, y_next)
        best_values.append(np.max(Y_sample))
    return X_sample, Y_sample, best_values, instant_regrets

# =============================================================================
# 5. Define Test Problems and Experiment Settings
# -----------------------------------------------------------------------------
problems = {
    "Rastrigin (3d)": {
         "func": neg_rast,
         "bounds": [(-5.12, 5.12)] * 3,
         "n_init": 15,
         "n_iter": 45
    },
    "Hartmann3 (3d)": {
         "func": hartmann3,
         "bounds": [(0, 1)] * 3,
         "n_init": 15,
         "n_iter": 45
    },
    "Hartmann4 (4d)": {
         "func": hartmann4,
         "bounds": [(0, 1)] * 4,
         "n_init": 20,
         "n_iter": 60
    },
    "Hartmann6 (6d)": {
         "func": hartmann6,
         "bounds": [(0, 1)] * 6,
         "n_init": 30,
         "n_iter": 90
    },
    "Branin (2d)": {
         "func": branin,
         "bounds": [(-5, 10), (0, 15)],
         "n_init": 10,
         "n_iter": 20
    },
    "Levy5 (5d)": {
         "func": levy5,
         "bounds": [(-10, 10)] * 5,
         "n_init": 25,
         "n_iter": 50
    }
}

num_experiments = 20  # Number of experiments per method per problem
methods_list = ['grid', 'quasi', 'nelder', 'CG']

results = {}  # To store raw experiment results

for prob_name, prob in problems.items():
    func = prob["func"]
    bounds_prob = prob["bounds"]
    n_init = prob["n_init"]
    n_iter = prob["n_iter"]
    d = len(bounds_prob)
    lower = np.array([b[0] for b in bounds_prob])
    upper = np.array([b[1] for b in bounds_prob])
    results[prob_name] = {}
    for method in methods_list:
        curves = []       # best-values curve per experiment
        cum_regrets = []  # cumulative regret curve per experiment
        times = []        # computation times per experiment
        for exp in range(num_experiments):
            print(f"Problem: {prob_name}, Method: {method}, Exp: {exp}")
            seed = 20252026 + exp
            sampler = qmc.Sobol(d, scramble=True, seed=seed)
            X_init_unit = sampler.random(n_init)
            X_init = qmc.scale(X_init_unit, lower, upper)
            Y_init = np.array([func(x) for x in X_init])
            start_time = time.perf_counter()
            _, _, best_values, instant_regrets = bayesian_optimization(
                func, bounds_prob, n_init=n_init, n_iter=n_iter,
                inner_optimizer=method, X_init=X_init, Y_init=Y_init)
            end_time = time.perf_counter()
            comp_time = end_time - start_time
            cum_regret = np.cumsum(instant_regrets)
            curves.append(best_values)
            cum_regrets.append(cum_regret)
            times.append(comp_time)
        results[prob_name][method] = {
            'curves': curves,
            'cum_regret': cum_regrets,
            'times': times
        }

# =============================================================================
# 6b. Compute Averages and Standard Deviations (including final outcome)
# -----------------------------------------------------------------------------
all_results = {}
final_outcome = []  # Table for final best value and its std.
for prob_name, prob_results in results.items():
    avg_curves = {}
    avg_cum_regret = {}
    std_cum_regret = {}
    avg_times = {}
    for method in methods_list:
        curves = np.array(prob_results[method]['curves'])         # shape: (num_experiments, iterations)
        cum_regrets = np.array(prob_results[method]['cum_regret'])   # shape: (num_experiments, iterations)
        avg_curve = np.mean(curves, axis=0)
        avg_reg = np.mean(cum_regrets, axis=0)
        std_reg = np.std(cum_regrets, axis=0)
        avg_time = np.mean(prob_results[method]['times'])
        avg_curves[method] = avg_curve
        avg_cum_regret[method] = avg_reg
        std_cum_regret[method] = std_reg
        avg_times[method] = avg_time
        
        # Final outcome: last value of best-values curve from each experiment
        final_vals = curves[:, -1]
        mean_final = np.mean(final_vals)
        std_final = np.std(final_vals)
        final_outcome.append({
            "Problem": prob_name,
            "Method": method,
            "MeanFinalValue": mean_final,
            "StdFinalValue": std_final,
            "AvgTime": avg_time
        })
        print(f"Problem: {prob_name}, Method: {method}, Avg. Comp Time: {avg_time:.4f} sec, Final Best Value: {mean_final:.4f} ± {std_final:.4f}")
    all_results[prob_name] = {
        'avg_curves': avg_curves,
        'avg_cum_regret': avg_cum_regret,
        'std_cum_regret': std_cum_regret,
        'avg_times': avg_times
    }

# =============================================================================
# 7. Save Averaged Results to CSV Files (including Std for cumulative regret)
# -----------------------------------------------------------------------------
# Convergence data
rows_conv = []
for prob, data in all_results.items():
    for method, curve in data['avg_curves'].items():
        for i, val in enumerate(curve):
            rows_conv.append({
                "Problem": prob,
                "Method": method,
                "Iteration": i,
                "Regret": -val
            })
df_conv = pd.DataFrame(rows_conv)
df_conv.to_csv("convergence.csv", index=False)
print("Saved convergence data to convergence.csv")

# Cumulative regret with standard deviation
rows_cum = []
for prob, data in all_results.items():
    for method in methods_list:
        avg_reg = data['avg_cum_regret'][method]
        std_reg = data['std_cum_regret'][method]
        for i, (val, std_val) in enumerate(zip(avg_reg, std_reg)):
            rows_cum.append({
                "Problem": prob,
                "Method": method,
                "BO_Iteration": i + 1,
                "CumulativeRegret": val,
                "Std": std_val
            })
df_cum = pd.DataFrame(rows_cum)
df_cum.to_csv("cumulative_regret.csv", index=False)
print("Saved cumulative regret data to cumulative_regret.csv")

# Computation times
rows_time = []
for prob, data in all_results.items():
    for method, t in data['avg_times'].items():
        rows_time.append({
            "Problem": prob,
            "Method": method,
            "AvgTime": t
        })
df_time = pd.DataFrame(rows_time)
df_time.to_csv("computation_times.csv", index=False)
print("Saved computation times to computation_times.csv")

# Final outcome table (with final best value and its std)
df_final = pd.DataFrame(final_outcome)
df_final.to_csv("final_outcome.csv", index=False)
print("Saved final outcome data to final_outcome.csv")

# =============================================================================
# 8. Read CSV Files and Plot the Results
# -----------------------------------------------------------------------------
df_conv = pd.read_csv("convergence.csv")
df_cum = pd.read_csv("cumulative_regret.csv")
df_time = pd.read_csv("computation_times.csv")

# --- Convergence Plot ---
problems_unsorted = df_conv["Problem"].unique()
def extract_dim(prob_name):
    try:
        return int(prob_name.split('(')[-1].split('d')[0])
    except Exception as e:
        return 100
sorted_problems = sorted(problems_unsorted, key=extract_dim)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for ax, prob in zip(axes, sorted_problems):
    subset = df_conv[df_conv["Problem"] == prob]
    for method in subset["Method"].unique():
        method_data = subset[subset["Method"] == method]
        ax.plot(method_data["Iteration"], method_data["Regret"], label=method)
    title_text = prob.replace("Levy5", "levy")
    ax.set_title(title_text, fontsize=24)
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("Simple Regret", fontsize=20)
    ax.grid(True)
handles, labels = axes[0].get_legend_handles_labels()
new_labels = []
for label in labels:
    if label.lower() == "quasi":
        new_labels.append("L-BFGS-B")
    elif label.lower() == "nelder":
        new_labels.append("NM")
    elif label.lower() == "grid":
        new_labels.append("Uniform")
    else:
        new_labels.append(label)
fig.legend(handles, new_labels, loc="lower center", ncol=len(new_labels), fontsize=20)
plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
plt.savefig("combined_convergence_increasing_dimension.png", bbox_inches='tight')
plt.show()

# --- Cumulative Regret Plot with Standard Deviation ---
problems_unsorted = df_cum["Problem"].unique()
sorted_problems = sorted(problems_unsorted, key=extract_dim)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for ax, prob in zip(axes, sorted_problems):
    subset = df_cum[df_cum["Problem"] == prob]
    for method in subset["Method"].unique():
        method_data = subset[subset["Method"] == method]
        x = method_data["BO_Iteration"]
        y = method_data["CumulativeRegret"]
        yerr = method_data["Std"]
        ax.plot(x, y, label=method)
        # Use np.maximum to ensure the lower bound does not go below 0
        ax.fill_between(x, np.maximum(y - yerr, 0), y + yerr, alpha=0.2)
    title_text = prob.replace("Levy5", "levy")
    ax.set_title(title_text, fontsize=24)
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("Cumulative Regret", fontsize=20)
    ax.grid(True)
handles, labels = axes[0].get_legend_handles_labels()
new_labels = []
for label in labels:
    if label.lower() == "quasi":
        new_labels.append("L-BFGS-B")
    elif label.lower() == "nelder":
        new_labels.append("NM")
    elif label.lower() == "grid":
        new_labels.append("Uniform")
    else:
        new_labels.append(label)
fig.legend(handles, new_labels, loc="lower center", ncol=len(new_labels), fontsize=20)
plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
plt.savefig("combined_cumulative_regret_increasing_dimension.png", bbox_inches='tight')
plt.show()


# --- Computation Times Bar Plot ---
df_time["Problem"] = df_time["Problem"].str.replace("Levy5", "levy")
def extract_dim2(prob_name):
    try:
        return int(prob_name.split("(")[-1].split("d")[0])
    except Exception as e:
        return 100
problems_ordered = sorted(df_time["Problem"].unique(), key=extract_dim2)
pivot_df = df_time.pivot(index="Method", columns="Problem", values="AvgTime")
pivot_df = pivot_df[problems_ordered]
pivot_df.rename(index={"quasi": "L-BFGS-B", "nelder": "NM", "grid": "Uniform"}, inplace=True)
plt.figure(figsize=(10, 6))
pivot_df.plot(kind="bar", figsize=(10, 6))
plt.title("Average Computation Time", size=22)
plt.xlabel("Method", size=20)
plt.ylabel("Avg. Time (sec)", size=20)
plt.xticks(rotation=0, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16, loc="best")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("computation_times_ordered.png", bbox_inches='tight')
plt.show()

print("Bar plot has been generated based on the CSV files.")
