import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def H1(y, x):
    return norm.cdf(y - 0.8 * x - 1, loc=0, scale=1)

def H2(y, x):
    return norm.cdf(y + 2, loc=0, scale=1)

def reward(S):
    return np.minimum(2, S**2)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# ReLU Derivative
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

### [A1]
# ω(z)
def omega_A1(z):
    return z
def rho_A1(z):
    return -1
def phi_A1(z):
    return z**2 / 2
def psi_A1(z):
    return -z

### [C1]
def omega_C1(z, a=0, b=10):
    return a / (1 + np.exp(z)) + b * np.exp(z) / (1 + np.exp(z))
def rho_C1(z):
    return - np.exp(z) / (1 + np.exp(z))
def phi_C1(z, a=0, b=10):
    return (b-a) / (1 + np.exp(z)) + b * np.log(1 + np.exp(z))
def psi_C1(z):
    return -np.log(1 + np.exp(z))

# Cost function
def J(u, u1, u2, omega, Y, d_func, phi_func, psi_func):
    return np.mean(phi_func(u) + (d_func(Y) + 0.8*np.maximum(omega(u1), omega(u2))) * psi_func(u))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, gamma, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.A1_1 = np.random.normal(0, np.sqrt(1 / input_size), (hidden_size, input_size))
        self.B1_1 = np.zeros((hidden_size, 1))
        self.A2_1 = np.random.normal(0, np.sqrt(1 / hidden_size), (output_size, hidden_size))
        self.B2_1 = np.zeros((output_size, 1))

        # Initialize ADAM parameters
        self.m_A1_1 = np.zeros_like(self.A1_1)
        self.v_A1_1 = np.zeros_like(self.A1_1)
        self.m_B1_1 = np.zeros_like(self.B1_1)
        self.v_B1_1 = np.zeros_like(self.B1_1)

        self.m_A2_1 = np.zeros_like(self.A2_1)
        self.v_A2_1 = np.zeros_like(self.A2_1)
        self.m_B2_1 = np.zeros_like(self.B2_1)
        self.v_B2_1 = np.zeros_like(self.B2_1)

        self.A1_2 = np.random.normal(0, np.sqrt(1 / input_size), (hidden_size, input_size))
        self.B1_2 = np.zeros((hidden_size, 1))
        self.A2_2 = np.random.normal(0, np.sqrt(1 / hidden_size), (output_size, hidden_size))
        self.B2_2 = np.zeros((output_size, 1))

        self.m_A1_2 = np.zeros_like(self.A1_2)
        self.v_A1_2 = np.zeros_like(self.A1_2)
        self.m_B1_2 = np.zeros_like(self.B1_2)
        self.v_B1_2 = np.zeros_like(self.B1_2)

        self.m_A2_2 = np.zeros_like(self.A2_2)
        self.v_A2_2 = np.zeros_like(self.A2_2)
        self.m_B2_2 = np.zeros_like(self.B2_2)
        self.v_B2_2 = np.zeros_like(self.B2_2)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 
        self.gamma = gamma
        self.learning_rate = learning_rate

    def forward1(self, X):
        self.Z1_1 = relu(np.dot(self.A1_1, X) + self.B1_1)
        self.Y1 = np.dot(self.A2_1, self.Z1_1) + self.B2_1
        return self.Y1

    def forward2(self, X):
        self.Z1_2 = relu(np.dot(self.A1_2, X) + self.B1_2)
        self.Y2 = np.dot(self.A2_2, self.Z1_2) + self.B2_2
        return self.Y2
    
    def backward1(self, X, Y, d_func, omega_func, rho_func):
        # Compute forward pass
        u = self.forward1(X)
        u1 = self.forward1(Y)
        u2 = self.forward2(Y)

        om1 = omega_func(u1)
        om2 = omega_func(u2)

        # Compute gradient components
        d_loss = d_func(Y) + self.gamma*np.maximum(om1, om2)- omega_func(u)
        v2 = d_loss * rho_func(u)
        
        # Compute gradients for output layer
        d_A2_1 = np.dot(v2, self.Z1_1.T) / X.shape[1]
        d_B2_1 = np.mean(v2, axis=1, keepdims=True)

        # Compute gradients for hidden layer
        v1 = np.dot(self.A2_1.T, v2) * relu_derivative(np.dot(self.A1_1, X) + self.B1_1)
        d_A1_1 = np.dot(v1, X.T) / X.shape[1]
        d_B1_1 = np.mean(v1, axis=1, keepdims=True)

        # Update time step
        self.t += 1

        # Update weights and biases using ADAM
        for param, grad, m, v in [
            (self.A1_1, d_A1_1, self.m_A1_1, self.v_A1_1),
            (self.B1_1, d_B1_1, self.m_B1_1, self.v_B1_1),
            (self.A2_1, d_A2_1, self.m_A2_1, self.v_A2_1),
            (self.B2_1, d_B2_1, self.m_B2_1, self.v_B2_1)
        ]:
            m[:] = self.beta1 * m + (1 - self.beta1) * grad  
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Correct bias for moments
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def backward2(self, X, Y, d_func, omega_func, rho_func):
        # Compute forward pass
        u = self.forward2(X)
        u1 = self.forward1(Y)
        u2 = self.forward2(Y)

        om1 = omega_func(u1)
        om2 = omega_func(u2)

        # Compute gradient components
        d_loss = d_func(Y) + self.gamma*np.maximum(om1, om2)- omega_func(u)
        v2 = d_loss * rho_func(u)
        
        # Compute gradients for output layer
        d_A2_2 = np.dot(v2, self.Z1_2.T) / X.shape[1]
        d_B2_2 = np.mean(v2, axis=1, keepdims=True)

        # Compute gradients for hidden layer
        v1 = np.dot(self.A2_2.T, v2) * relu_derivative(np.dot(self.A1_2, X) + self.B1_2)
        d_A1_2 = np.dot(v1, X.T) / X.shape[1]
        d_B1_2 = np.mean(v1, axis=1, keepdims=True)

        # Update time step
        self.t += 1

        # Update weights and biases using ADAM
        for param, grad, m, v in [
            (self.A1_2, d_A1_2, self.m_A1_2, self.v_A1_2),
            (self.B1_2, d_B1_2, self.m_B1_2, self.v_B1_2),
            (self.A2_2, d_A2_2, self.m_A2_2, self.v_A2_2),
            (self.B2_2, d_B2_2, self.m_B2_2, self.v_B2_2)
        ]:
            m[:] = self.beta1 * m + (1 - self.beta1) * grad  
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Correct bias for moments
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def train(self, X1, X2, Y1, Y2, epochs, d_func, omega_func, rho_func, phi_func, psi_func, g_name, functions_name):
        losses1 = []
        losses2 = []
        print(f"\nTraining Neural Network for V1(S), V2(S) with {functions_name}.")
        print("=" * 50)

        for epoch in range(epochs):
            self.backward1(X1, Y1, d_func, omega_func, rho_func)
            self.backward2(X2, Y2, d_func, omega_func, rho_func)
            loss1 = J(self.forward1(X1), self.forward1(Y1), self.forward2(Y1), omega_func, Y1, d_func, phi_func, psi_func)
            loss2 = J(self.forward2(X2), self.forward1(Y2), self.forward2(Y2), omega_func, Y2, d_func, phi_func, psi_func)
            losses1.append(loss1)
            losses2.append(loss2)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss {functions_name}-1: {loss1:.4f} Loss {functions_name}-2: {loss2:.4f}")

        # Plot learning curve
        plt.plot(losses1, label=functions_name+"-1")
        plt.plot(losses2, label=functions_name+"-2")
        plt.xlabel("Number of Iterations")
        plt.title(f"Learning Curve for {g_name} with {functions_name}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    ### Numerical solution
    N = 1000
    gamma=0.8
    St = np.linspace(-10, 10, N)
    V1 = np.zeros(N)
    V2 = np.zeros(N)
    R = np.array([reward(s) for s in St])

    F1 = np.zeros((N, N))
    F2 = np.zeros((N, N))
    for j in range(N):
        F1[j][0] = 0.5 * (H1(St[1], St[j]) - H1(St[0], St[j]))
        F1[j][-1] = 0.5 * (H1(St[-1], St[j]) - H1(St[-2], St[j]))

        F2[j][0] = 0.5 * (H2(St[1], St[j]) - H2(St[0], St[j]))
        F2[j][-1] = 0.5 * (H2(St[-1], St[j]) - H2(St[-2], St[j]))

        for i in range(2, N):
            F1[j][i - 1] = 0.5 * (H1(St[i], St[j]) - H1(St[i - 2], St[j]))
            F2[j][i - 1] = 0.5 * (H2(St[i], St[j]) - H2(St[i - 2], St[j]))

    for _ in range(100000):
        V1_new = np.dot(F1, (R + gamma * np.maximum(V1, V2)))
        V2_new = np.dot(F2, (R + gamma * np.maximum(V1, V2)))
        V1 = V1_new
        V2 = V2_new

    # Generate actions
    actions = np.zeros(N)
    for i in range(N):
        actions[i] = 1 if np.random.uniform(0, 1) > 0.5 else 2
    
    # Generate States
    States = np.zeros(N+1)
    States[0] = np.random.normal(0, 1)

    for t in range(N):
        Wt = np.random.normal(0, 1)
        if actions[t] == 1:
            States[t+1] = 0.8 * States[t] + 1.0 + Wt
        else:
            States[t+1] = -2.0 + Wt

    # Separate the data into two sets based on the action
    set_action_1 = [(States[t], States[t + 1]) for t in range(N) if actions[t] == 1]
    set_action_2 = [(States[t], States[t + 1]) for t in range(N) if actions[t] == 2]

    # Convert to numpy arrays
    set_action_1 = np.array(set_action_1)
    set_action_2 = np.array(set_action_2)
    
    # Training data for v1(S)
    X1 = set_action_1[:, 0].reshape(1, -1)
    Y1 = set_action_1[:, 1].reshape(1, -1)
    # Training data for v2(S)
    X2 = set_action_2[:, 0].reshape(1, -1)
    Y2 = set_action_2[:, 1].reshape(1, -1)

    # Neural Network parameters
    input_size = 1
    hidden_size = 100
    output_size = 1
    learning_rate = 0.001
    epochs = 10000

    # Create and train the neural networks
    nn_A1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, gamma) 
    nn_A1.train(X1, X2, Y1, Y2, epochs, d_func=reward, omega_func=omega_A1, rho_func=rho_A1, phi_func=phi_A1, psi_func=psi_A1, g_name=r"$u(X,θ^1_o)$", functions_name="[A1]")
    nn_A1_pred_V1 = np.array(omega_A1([nn_A1.forward1(np.array([[x]])) for x in St]))
    nn_A1_pred_V2 = np.array(omega_A1([nn_A1.forward2(np.array([[x]])) for x in St]))

    nn_C1 = NeuralNetwork(input_size, hidden_size, output_size, learning_rate, gamma) 
    nn_C1.train(X1, X2, Y1, Y2, epochs, d_func=reward, omega_func=omega_C1, rho_func=rho_C1, phi_func=phi_C1, psi_func=psi_C1, g_name=r"$u(X,θ^2_o)$", functions_name="[C1]")
    nn_C1_pred_V1 = np.array(omega_C1([nn_C1.forward1(np.array([[x]])) for x in St]))
    nn_C1_pred_V2 = np.array(omega_C1([nn_C1.forward2(np.array([[x]])) for x in St]))

    plt.figure(figsize=(10, 6))
    plt.plot(St, V1, label="Numerical", color="black")
    plt.plot(St, nn_A1_pred_V1.flatten(), label="[A1]", color="blue")
    plt.plot(St, nn_C1_pred_V1.flatten(), label="[C1]", color="red")
    plt.xlim(-10, 10)
    plt.ylim(7, 11)
    plt.title(r"$Conditional\ Expectation\ v_1(X)$")
    plt.xlabel(r'$\mathcal{X}$', fontsize=16)
    plt.ylabel(r'$\mathbb{E}^1_{S_{t+1}} \!\left[ R(S_{t+1}) + \max \{V_1(S_{t+1}),V_2(S_{t+1})\} \,\middle|\, S_t = X \right]$',fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(St, V2, label="Numerical", color="black")
    plt.plot(St, nn_A1_pred_V2.flatten(), label="[A1]", color="blue")
    plt.plot(St, nn_C1_pred_V2.flatten(), label="[C1]", color="red")
    plt.xlim(-10, 10)
    plt.ylim(7, 11)
    plt.title(r"$Conditional\ Expectation\ v_2(X)$")
    plt.xlabel(r'$\mathcal{X}$', fontsize=16)
    plt.ylabel(r'$\mathbb{E}^2_{S_{t+1}} \!\left[ R(S_{t+1}) + \max \{V_1(S_{t+1}),V_2(S_{t+1})\} \,\middle|\, S_t = X \right]$',fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(St, V1, label="Numerical-1", color="black")
    plt.plot(St, V2, label="Numerical-2", color="grey")
    plt.plot(St, nn_A1_pred_V1.flatten(), label="[A1]-1")
    plt.plot(St, nn_C1_pred_V1.flatten(), label="[C1]-1")
    plt.plot(St, nn_A1_pred_V2.flatten(), label="[A1]-2")
    plt.plot(St, nn_C1_pred_V2.flatten(), label="[C1]-2")
    plt.xlim(-10, 10)
    plt.ylim(6, 11)
    plt.title(r"$Conditional\ Expectations\ v_1(X),v_2(X)$")
    plt.xlabel(r'$\mathcal{S}$', fontsize=16)
    plt.ylabel(r'$\mathbb{E}^j_{S_{t+1}} \!\left[ R(S_{t+1}) + \max \{V_1(S_{t+1}),V_2(S_{t+1})\} \,\middle|\, S_t = X \right]$',fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.show()

        # Optimal Actions
    opt_actions = np.zeros(N)
    A1_actions = np.zeros(N)
    C1_actions = np.zeros(N)

    for t in range(N):
        opt_actions[t] = 1 if V1[t] > V2[t] else 2
        A1_actions[t] = 1 if omega_A1(nn_A1.forward1(St[t])) > omega_A1(nn_A1.forward2(St[t])) else 2
        C1_actions[t] = 1 if omega_C1(nn_C1.forward1(St[t])) > omega_C1(nn_C1.forward2(St[t])) else 2

    plt.figure(figsize=(10, 6))
    plt.plot(St, opt_actions, label="Optimal")
    plt.plot(St, A1_actions, label="[A1]")
    plt.plot(St, C1_actions, label="[C1]")
    plt.xlim(St.min(), St.max())
    plt.ylim(0, 3)
    plt.xlabel(r'$\mathcal{S}$', fontsize=16)
    plt.ylabel(r"$Action\ Policy\ \{a_t\}$")
    plt.title("Optimal and Approximately optimal action policy")
    plt.legend()
    plt.grid(True)
    plt.show()