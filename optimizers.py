import numpy as np

map_step = .005


def get_possible_steps(rmap, theta):
    try:
        step_north = rmap.get_cost(theta[0] + map_step, theta[1])
        step_south = rmap.get_cost(theta[0] - map_step, theta[1])
        step_east = rmap.get_cost(theta[0], theta[1] + map_step)
        step_west = rmap.get_cost(theta[0], theta[1] - map_step)
    except IndexError:
        print('The boundary of elevation map has been reached')
        return None

    return step_north, step_east, step_south, step_west


def calculate_gradient(rmap, theta, j_history, n_iter):
    cost = rmap.get_cost(*theta)
    elevation = rmap.get_elevation(*theta)
    j_history[n_iter] = [elevation, theta[0], theta[1]]

    steps = get_possible_steps(rmap, theta)

    if cost <= 0 or steps is None:
        return None

    lat_slope = steps[0] / steps[2] - 1
    lon_slope = steps[1] / steps[3] - 1
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


def simulated_annealing(map, theta):
    pass


def stochastic_hill_climb(map, theta):
    pass


def tabu_search(map, theta):
    pass


def particle_swarm(map, theta):
    pass
