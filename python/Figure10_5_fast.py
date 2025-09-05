# Author: Kenji Kashima
# Date  : 2025/05/24
# Figure10_5.py と同じ結果．計算時間短縮

import numpy as np
import matplotlib.pyplot as plt
import config

np.random.seed(23)

directions = [(1,0), (-1,0), (0,1), (0,-1)]  # Right, Left, Up, Down

def init_value( ):
    # Define value matrix
    value = np.zeros((grid_size + 1, grid_size + 1))

    # Fill value matrix with the given equation
    for i in range(grid_size + 1):
        for j in range(grid_size + 1 ):
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

def all_agents_locally_optimal(state, value, sensor_range):
    IDn = state.shape[0]
    grid_size = value.shape[0]

    for ID in range(IDn):
        x0, y0 = state[ID, 0], state[ID, 1]
        base_value = sum_value_near_agents(state, sensor_range, x0, y0, value)

        for dx, dy in directions:
            proposed = state.copy()
            proposed[ID, 0] = np.clip(x0 + dx, 0, grid_size)
            proposed[ID, 1] = np.clip(y0 + dy, 0, grid_size)

            new_value = sum_value_near_agents(proposed, sensor_range, x0, y0, value)

            if new_value > base_value:
                return False  # 少なくとも1方向に改善できる → 局所最適ではない

    return True  # 全方向に対して改善できない → 局所最適

# Define the simulation function
def simulation(value, ini_state, Tmax, deterministic):
    """
    Runs a simulation of IDn agents moving in a 2D grid based on local value estimations.

    Parameters:
        value (ndarray): (101, 101) value map.
        Tmax (int): maximum simulation steps.
        deterministic (bool): if True, always move if estimated value improves.

    Returns:
        ini_state (ndarray): initial agent positions.
        state (ndarray): current agent positions.
        x_list (ndarray): trajectory of x-coordinates (shape: [IDn, t]).
        y_list (ndarray): trajectory of y-coordinates (shape: [IDn, t]).
    """

    x_list = np.zeros((IDn, Tmax))
    y_list = np.zeros((IDn, Tmax))

    state = ini_state.copy()

    for t in range(Tmax):
        x_list[:, t] = state[:, 0]
        y_list[:, t] = state[:, 1]

        # Randomly select an agent and movement direction
        agent_id = np.random.randint(IDn)
        direction = directions[np.random.randint(4)]
        candidate_state = state.copy()

        # Apply movement in chosen direction
        candidate_state[agent_id, 0] = np.clip(candidate_state[agent_id, 0] + direction[0], 0, grid_size)
        candidate_state[agent_id, 1] = np.clip(candidate_state[agent_id, 1] + direction[1], 0, grid_size)

        # Evaluate local values
        x, y = state[agent_id, 0], state[agent_id, 1]
        current_value = sum_value_near_agents(state, sensor_range, x, y, value)
        candidate_value = sum_value_near_agents(candidate_state, sensor_range, x, y, value)

        if deterministic:
            if candidate_value > current_value:
                state = candidate_state
                print(f"Step {t}: Agent {agent_id} moved (deterministic)")

                if all_agents_locally_optimal(state, value, sensor_range):
                    x_list = x_list[:, :t+1]
                    y_list = y_list[:, :t+1]
                    print(f"All agents are locally optimal at step {t}. Simulation terminated.")
                    break
        else:
            prob_move = 1 / (1 + np.exp(beta * (current_value - candidate_value)))
            if np.random.rand() < prob_move:
                state = candidate_state
                print(f"Step {t}: Agent {agent_id} moved (stochastic)")

    return state, x_list[:, :t+1], y_list[:, :t+1]

def sum_value_near_agents(state, sensor_range, X, Y, value):
    """
    Vectorized version. Returns the sum of value[I, J] around (X, Y)
    where at least one agent is within 'sensor_range' of (I, J).
    """
    evaluation_range = int(sensor_range + 3)
    X, Y = int(X), int(Y)

    # 定義域の切り出し（境界考慮）
    I_min = max(1, X - evaluation_range)
    I_max = min(grid_size, X + evaluation_range) + 1
    J_min = max(1, Y - evaluation_range)
    J_max = min(grid_size, Y + evaluation_range) + 1

    # 対象グリッド座標をベクトルで生成
    I, J = np.meshgrid(np.arange(I_min, I_max), np.arange(J_min, J_max), indexing='ij')
    I_flat = I.ravel()
    J_flat = J.ravel()
    grid_points = np.stack([I_flat, J_flat], axis=1)  # shape: (M, 2)

    # state: shape (N, 2)
    # grid_points: shape (M, 2)
    # → 距離行列 (N, M)
    dists_sq = np.sum((state[:, None, :] - grid_points[None, :, :])**2, axis=2)  # (N, M)
    is_close = np.any(dists_sq < sensor_range**2, axis=0)  # shape (M,)

    # 対象 value を抽出して合計
    values_flat = value[I_flat, J_flat]
    return np.sum(values_flat[is_close])


def Figure10_5a(value):
    figsize = config.global_config(type= 1)
    # Figure 10.5(a): Contour plot of the value function
    plt.figure(figsize=figsize)

    plt.contourf(value.T, 20)
    plt.colorbar()
    plt.xlim([0, grid_size])
    plt.ylim([0, grid_size])
    plt.tight_layout()
    plt.savefig("./Figure10_5a.pdf")
    plt.show()

def Figure10_5b(value, Tmax=20000):
    figsize = config.global_config(type= 1)
    # Figure 10.5(b): Run simulation with deterministic=False
    state, x_list, y_list = simulation(value, ini_state, Tmax, False)
    plt.figure(figsize=figsize)
    plt.plot(ini_state[:, 0], ini_state[:, 1], '*', markersize=10)
    plt.plot(state[:, 0], state[:, 1], 'o', markersize=10)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(IDn):
        plt.plot(x_list[i, :], y_list[i, :])
        plt.plot(10 * np.sin(t) + state[i, 0], 10 * np.cos(t) + state[i, 1])
    plt.grid(True)
    plt.xlim([0,grid_size])
    plt.ylim([0,grid_size])
    plt.tight_layout()
    plt.savefig("./Figure10_5b.pdf")
    plt.show()

def Figure10_5c(value, Tmax=20000):
    figsize = config.global_config(type= 1)
    # Figure 10.5(c): Run simulation with deterministic=True
    state, x_list, y_list = simulation(value, ini_state, Tmax, True)
    plt.figure(figsize=figsize)
    plt.plot(ini_state[:, 0], ini_state[:, 1], '*', markersize=10)
    plt.plot(state[:, 0], state[:, 1], 'o', markersize=10)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(IDn):
        plt.plot(x_list[i, :], y_list[i, :])
        plt.plot(10 * np.sin(t) + state[i, 0], 10 * np.cos(t) + state[i, 1])
    plt.grid(True)
    plt.xlim([0,grid_size])
    plt.ylim([0,grid_size])
    plt.tight_layout()
    plt.savefig("./Figure10_5c.pdf")
    plt.show()

if __name__ == '__main__':
    grid_size = 100
    IDn = 12
    sensor_range = 8
    beta = 1
    ini_state = np.ones((IDn, 2)) * 10
    
    value = init_value( )
    Figure10_5a(value)
    Figure10_5b(value)
    Figure10_5c(value)
