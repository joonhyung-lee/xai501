import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
"""
    Let us limit the input sapce x in the range of [-1, 1]
    Generate 100 i.i.d. sample points, {(x_n, t_n)}_{n=1}^{100}, from a function t_n = cos(2 * pi * x_n) + sin(pi * x_n) + epsilon_n, where epsilon_n ~ N(0, 1/beta) and beta = 11.1 fixed.
"""
def generate_data(n_samples, beta):
    x = np.linspace(-1, 1, n_samples)
    # x = np.random.uniform(-1, 1, n)
    noise = np.random.normal(0, 1/np.sqrt(beta), n_samples)
    t = np.cos(2*np.pi*x) + np.sin(np.pi*x) + noise
    return x, t

# Design matrix for polynomial regression
"""
    Use a 9-th order polynomial function defined as follows: y(x, w) = w_0 + w_1 * x + w_2 * x^2 + ... + w_9 * x^9
"""
def model(x, degree):
    Phi = np.zeros((len(x), degree+1))
    for i in range(degree+1):
        Phi[:, i] = x**i
    return Phi

# 1. Maximum Likelihood Estimation (MLE)
def mle(x, t, degree):
    Phi = model(x, degree)
    w_mle = np.linalg.pinv(Phi) @ t
    return w_mle

# 2. Maximum a Posteriori (MAP)
"""
    For both MAP and Bayesian, use a prior distribution over w as p(w|a) = N(0, a^(-1)I), where a = 5e-3 fixed.
"""
def map_estimation(x, t, degree, alpha):
    Phi = model(x, degree)
    S_N_inv = alpha*np.eye(degree+1) + Phi.T @ Phi
    S_N = np.linalg.inv(S_N_inv)
    m_N = S_N @ Phi.T @ t
    return m_N

# 3. Bayesian Linear Regression
def bayesian_regression(x, t, degree, alpha, beta, x_test):
    Phi = model(x, degree)
    S_N_inv = alpha*np.eye(degree+1) + beta*Phi.T @ Phi
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta*S_N @ Phi.T @ t
    
    Phi_test = model(x_test, degree)
    y_test = Phi_test @ m_N
    y_var = 1/beta + np.sum(Phi_test @ S_N * Phi_test, axis=1)
    y_std = np.sqrt(y_var)
    
    return y_test, y_std

# Main function
def main(x_test):
    n_samples = 100
    degree = 9
    alpha = 5e-3
    beta = 11.1
    x_train, t_train = generate_data(n_samples, beta)

    # MLE
    w_mle = mle(x_train, t_train, degree)
    y_mle = model(x_test, degree) @ w_mle

    # MAP
    w_map = map_estimation(x_train, t_train, degree, alpha)
    y_map = model(x_test, degree) @ w_map

    # Bayesian
    y_bayes, y_std = bayesian_regression(x_train, t_train, degree, alpha, beta, x_test)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].scatter(x_train, t_train, color='grey', label='Training Data')
    axs[0].plot(x_test, y_mle, color='red', label='MLE')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')
    axs[0].set_title('MLE')
    axs[0].legend()
    
    axs[1].scatter(x_train, t_train, color='grey', label='Training Data')
    axs[1].plot(x_test, y_map, color='blue', label='MAP')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')
    axs[1].set_title('MAP')
    axs[1].legend()
    
    axs[2].scatter(x_train, t_train, color='grey', label='Training Data')
    axs[2].plot(x_test, y_bayes, color='green', label='Bayesian')
    axs[2].fill_between(x_test, y_bayes-y_std, y_bayes+y_std, color='lightgreen', alpha=0.2, label='Bayesian $\pm~1$ std')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('t')
    axs[2].set_title('Bayesian')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, t_train, color='grey', label='Training Data')
    plt.plot(x_test, y_mle, color='red', label='MLE')
    plt.plot(x_test, y_map, color='blue', label='MAP')
    plt.plot(x_test, y_bayes, color='green', label='Bayesian')
    plt.fill_between(x_test, y_bayes-y_std, y_bayes+y_std, color='lightgreen', alpha=0.2, label='Bayesian $\pm$1 std')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Regression')
    plt.legend()
    plt.show()
    
    print(f"Predicted Target value \nx={x_test[0]:.2f}: {y_bayes[0]:.4f} $\pm$ {y_std[0]:.4f}")

if __name__ == "__main__":
    x_test = np.linspace(-1, 1, 100)  # Test points
    main(x_test)