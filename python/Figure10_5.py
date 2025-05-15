# Author: Kenji Kashima
# Date  : 2025/04/01

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(23)
import sys
sys.path.append("./")
import config

def init_value():
    # Define grid size and initial value matrix
    grid_size = 101
    value = np.zeros((grid_size, grid_size))

    # Fill value matrix with the given equation
    for i in range(grid_size):
        for j in range(grid_size):
            value[i, j] = (1.5 * np.exp(-((i-20)**2)/1000 - ((j-20)**2)/1000) +
                        np.exp(-(i-20)**2/1000 - (j-90)**2/500) +
                        np.exp(-(i-40)**2/1000 - (j-50)**2/500) +
                        np.exp(-(i-70)**2/300 - (j-70)**2/500) +
                        1.5 * np.exp(-(i-80)**2/300 - (j-40)**2/500) +
                        np.exp(-(i-50)**2/800 - (j-50)**2/800) +
                        1.2 * np.exp(-(i-80)**2/200 - (j-20)**2/200) +
                        np.exp(-(i-90)**2/200 - (j-10)**2/200))
    value /= 10
    return value



# Define the simulation function
def simulation(value, Tmax, deterministic):
    IDn = 12
    sensor_range = 8
    beta = 1
    x1_list = np.zeros((IDn, Tmax))
    x2_list = np.zeros((IDn, Tmax))
    ini_state = np.ones((IDn, 2)) * 10  # Initial positions (x1, x2)
    state = ini_state.copy()

    for t in range(Tmax):
        x1_list[:, t] = state[:, 0]
        x2_list[:, t] = state[:, 1]
        
        # Estimation and random move
        state_hat = state.copy()
        select_ID = np.random.randint(IDn)
        select_direction = np.random.randint(4)
        
        if select_direction == 0:  # Right
            state_hat[select_ID, 0] = min(100, state_hat[select_ID, 0] + 1)
        elif select_direction == 1:  # Left
            state_hat[select_ID, 0] = max(0, state_hat[select_ID, 0] - 1)
        elif select_direction == 2:  # Up
            state_hat[select_ID, 1] = min(100, state_hat[select_ID, 1] + 1)
        elif select_direction == 3:  # Down
            state_hat[select_ID, 1] = max(0, state_hat[select_ID, 1] - 1)

        X1, X2 = state[select_ID, 0], state[select_ID, 1]
        l_x = weighted_state_position_value(state, sensor_range, X1, X2, value)
        l_x_hat = weighted_state_position_value(state_hat, sensor_range, X1, X2, value)

        l_all = [l_x, l_x_hat]
        move = soft_max(l_all, beta, deterministic)

        if move > 1:
            state = state_hat
        print(t,"/",Tmax)

    return ini_state, state, x1_list, x2_list

# Helper functions
def weighted_state_position_value(state, distance, X1, X2, value):
    weighted_value = 0
    X1, X2 = int(X1), int(X2)
    for I in range(max(1, X1 - 11), min(100, X1 + 11)):
        for J in range(max(1, X2 - 11), min(100, X2 + 11)):
            weighted_value += value[I, J] * state_position_value(state, I, J, distance)
    return weighted_value

def state_position_value(state, X1, X2, distance):
    value = 0
    IDn = state.shape[0]
    for ID in range(IDn):
        if (state[ID, 0] - X1) ** 2 + (state[ID, 1] - X2) ** 2 < distance ** 2:
            value = min(1, value + 1)
    return value

def soft_max(l_all, beta, deterministic):
    l_all = np.exp(beta * np.array(l_all))
    threshold = l_all[0] / np.sum(l_all)
    if deterministic:
        return 1 if threshold >= 0.5 else 2
    else:
        return 1 if np.random.rand() <= threshold else 2

def Figure10_5a(value):
    figsize = config.global_config(type= 1)
    # Figure 10.5(a): Contour plot of the value function
    plt.figure(figsize=figsize)

    plt.contourf(value.T, 20)
    plt.colorbar()
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig("./figures/Figure10_5a.pdf")
    plt.show()

def Figure10_5b(value, Tmax=20000):
    figsize = config.global_config(type= 1)
    # Figure 10.5(b): Run simulation with deterministic=False
    ini_state, state, x1_list, x2_list = simulation(value, Tmax, False)
    plt.figure(figsize=figsize)
    plt.plot(ini_state[:, 0], ini_state[:, 1], '*', markersize=10)
    plt.plot(state[:, 0], state[:, 1], 'o', markersize=10)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(12):
        plt.plot(x1_list[i, :], x2_list[i, :])
        plt.plot(10 * np.sin(t) + state[i, 0], 10 * np.cos(t) + state[i, 1])
    plt.grid(True)
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.tight_layout()
    plt.savefig("./figures/Figure10_5b.pdf")
    plt.show()

def Figure10_5c(value, Tmax=20000):
    figsize = config.global_config(type= 1)
    # Figure 10.5(c): Run simulation with deterministic=True
    ini_state, state, x1_list, x2_list = simulation(value, Tmax, True)
    plt.figure(figsize=figsize)
    plt.plot(ini_state[:, 0], ini_state[:, 1], '*', markersize=10)
    plt.plot(state[:, 0], state[:, 1], 'o', markersize=10)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(12):
        plt.plot(x1_list[i, :], x2_list[i, :])
        plt.plot(10 * np.sin(t) + state[i, 0], 10 * np.cos(t) + state[i, 1])
    plt.grid(True)
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.tight_layout()
    plt.savefig("./figures/Figure10_5c.pdf")
    plt.show()

if __name__ == '__main__':
    value = init_value()
    Figure10_5a(value)
    Figure10_5b(value)
    Figure10_5c(value)
