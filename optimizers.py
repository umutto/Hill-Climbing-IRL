import random
import numpy as np


def get_step_costs(rmap, steps):
    return [rmap.get_cost(*s) for s in steps]


def get_possible_steps(theta, step_size=.005):
    try:
        step_north = (theta[0] + step_size, theta[1])
        step_south = (theta[0] - step_size, theta[1])
        step_east = (theta[0], theta[1] + step_size)
        step_west = (theta[0], theta[1] - step_size)
    except IndexError:
        print('The boundary of elevation map has been reached')
        return None

    return step_north, step_east, step_south, step_west


def calculate_gradient(rmap, theta, j_history, n_iter):
    cost = rmap.get_cost(*theta)
    elevation = rmap.get_elevation(*theta)
    j_history[n_iter] = [elevation, theta[0], theta[1]]

    step_costs = get_step_costs(rmap, get_possible_steps(theta))

    if cost <= 0 or step_costs is None:
        return None

    lat_slope = step_costs[0] / step_costs[2] - 1
    lon_slope = step_costs[1] / step_costs[3] - 1
    print(f'Elevation at {theta} is {elevation}')

    return np.array((lat_slope, lon_slope))


def gradient_descent(map, theta, alpha=.01, num_iters=10000):
    j_history = np.zeros(shape=(num_iters, 3))

    for i in range(num_iters):
        slope = calculate_gradient(map, theta, j_history, i)
        if slope is None:
            break

        step = -alpha * slope
        print(f'({i}/{num_iters}): Update is {step}')
        theta += step

    return theta, j_history[:i]


def gradient_descent_w_momentum(map, theta, alpha=.01, mu=.99, num_iters=10000):
    j_history = np.zeros(shape=(num_iters, 3))
    velocity = np.zeros_like(theta)

    for i in range(num_iters):
        slope = calculate_gradient(map, theta, j_history, i)
        if slope is None:
            break

        velocity = mu * velocity - alpha * slope
        print(f'({i}/{num_iters}): Update is {velocity}')

        theta += velocity

    return theta, j_history[:i]


def gradient_descent_w_nesterov(map, theta, alpha=.01, mu=.99, num_iters=10000):
    j_history = np.zeros(shape=(num_iters, 3))
    velocity = np.zeros_like(theta)
    v_prev = np.zeros_like(theta)

    for i in range(num_iters):
        slope = calculate_gradient(map, theta, j_history, i)
        if slope is None:
            break

        v_prev = np.copy(velocity)
        velocity = mu * velocity - alpha * slope

        step = -mu * v_prev + (1 + mu) * velocity
        print(f'({i}/{num_iters}): Update is {step}')

        theta += step

    return theta, j_history[:i]


def adagrad(map, theta, alpha=.01, epsilon=1e-8, num_iters=10000):
    j_history = np.zeros(shape=(num_iters, 3))
    cache = np.zeros_like(theta)

    for i in range(num_iters):
        slope = calculate_gradient(map, theta, j_history, i)
        if slope is None:
            break

        cache += slope**2

        step = -alpha * slope / (np.sqrt(cache) + epsilon)
        print(f'({i}/{num_iters}): Update is {step}')

        theta += step

    return theta, j_history[:i]


def RMSprop(map, theta, alpha=.001, epsilon=1e-8, decay_rate=.99, num_iters=10000):
    j_history = np.zeros(shape=(num_iters, 3))
    cache = np.zeros_like(theta)

    for i in range(num_iters):
        slope = calculate_gradient(map, theta, j_history, i)
        if slope is None:
            break

        cache = decay_rate * cache + (1 - decay_rate) * slope**2

        step = -alpha * slope / (np.sqrt(cache) + epsilon)
        print(f'({i}/{num_iters}): Update is {step}')

        theta += step

    return theta, j_history[:i]


def adam(map, theta, alpha=.001, epsilon=1e-8, beta1=.9,
         beta2=.999, num_iters=10000):
    j_history = np.zeros(shape=(num_iters, 3))
    m, v = np.zeros_like(theta), np.zeros_like(theta)

    # bias correction
    mt, vt = np.zeros_like(theta), np.zeros_like(theta)

    for i in range(num_iters):
        slope = calculate_gradient(map, theta, j_history, i)
        if slope is None:
            break

        t = i + 1

        m = beta1 * m + (1 - beta1) * slope
        mt = m / (1 - beta1**t)
        v = beta2 * v + (1 - beta2) * slope**2
        vt = v / (1 - beta2**t)

        step = -alpha * mt / (np.sqrt(vt) + epsilon)
        print(f'({i}/{num_iters}): Update is {step}')

        theta += step

    return theta, j_history[:i]


def genetic_alg(map, theta):
    pass


def simulated_annealing(map, theta, alpha=.99, temp=1,
                        min_temp=1e-6, num_iters=10000):
    cost = map.get_cost(*theta)
    j_history = np.zeros(shape=(num_iters, 3))
    j_history[0] = [map.get_elevation(*theta), theta[0], theta[1]]

    def prob(c, n_c, t):
        p = np.e**((c - n_c) / t)
        if p >= np.random.random():
            return True
        return False

    for i in range(1, num_iters):
        if temp < min_temp:
            break

        for j in range(50):
            steps = get_possible_steps(theta)
            step_costs = get_step_costs(map, steps)

            step, step_cost = random.choice(list(zip(steps, step_costs)))
            if prob(cost, step_cost, temp):
                theta = step
                cost = step_cost

        elevation = map.get_elevation(*theta)
        j_history[i] = [elevation, theta[0], theta[1]]

        print(f'Elevation at {theta} is {elevation}')
        print(f'({i}/{num_iters}): Update is {theta}')

        temp *= alpha

    return theta, j_history[:i]


def stochastic_hill_climb(map, theta):
    pass


def tabu_search(map, theta):
    pass


def particle_swarm(map, theta):
    pass
