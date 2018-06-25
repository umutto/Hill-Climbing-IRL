import os
import json
import csv

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from rastermap import RasterMap
import optimizers as opt

args = json.load(open('params.json'))

current_map = RasterMap(args['tif'])

methods = {
    'Gradient Descent': {'fun': opt.gradient_descent, 'color': '#FF0000'},
    'Momentum': {'fun': opt.gradient_descent_w_momentum, 'color': '#009933'},
    'NAG': {'fun': opt.gradient_descent_w_nesterov, 'color': '#9900FF'},
    'Adagrad': {'fun': opt.adagrad, 'color': '#0066FF'},
    'RMSprop': {'fun': opt.RMSprop, 'color': '#000000'},
    'Adam': {'fun': opt.adam, 'color': '#FFFF00'},
    'Simulated Annealing': {'fun': opt.simulated_annealing, 'color': '#ED7504'},
    'Stochastic Hill Climb': {'fun': opt.stochastic_hill_climb, 'color': '#F442C5'},
    'Tabu Search': {'fun': opt.tabu_search, 'color': '#56FCFF'}
}

# clean before running the experiment
for csv_file in os.listdir('outputs/'):
    os.remove(f'outputs/{csv_file}')

fig = plt.figure()
for k, v in methods.items():
    print(f"\n{'-'*10} {k} {'-'*10}")
    theta, j_history = v['fun'](current_map, np.array([args['center']['lat'],
                                                       args['center']['lng']]),
                                num_iters=args['iters'])

    with open(f'{args["output"]}{"-".join(k.strip().split())}.csv',
              'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([v['color']])
        for weight in j_history:
            writer.writerow([weight[1], weight[2]])

    plt.plot(range(j_history.shape[0]), j_history[:, 0], label=k, c=v['color'])

plt.xlabel('Iterations')
plt.ylabel('Elevation')
plt.title('Hill Climbing Algorithms')
plt.legend()
plt.tight_layout()

# save the plot
fig.savefig('Cost_Plot.png', dpi=fig.dpi)
plt.show()
