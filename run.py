import json
import csv

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from rastermap import RasterMap
from optimizers import (gradient_descent, gradient_descent_w_momentum,
                        gradient_descent_w_nesterov, adagrad, RMSprop, adam)

args = json.load(open('params.json'))

current_map = RasterMap(args['tif'])

methods = {
    'Gradient Descent': {'fun': gradient_descent, 'color': '#FF0000'},
    'Momentum': {'fun': gradient_descent_w_momentum, 'color': '#009933'},
    'NAG': {'fun': gradient_descent_w_nesterov, 'color': '#9900FF'},
    'Adagrad': {'fun': adagrad, 'color': '#0066FF'},
    'RMSprop': {'fun': RMSprop, 'color': '#000000'},
    'Adam': {'fun': adam, 'color': '#FFFF00'}
}

for k, v in methods.items():
    theta, j_history = v['fun'](current_map, np.array([args['center']['lat'],
                                                       args['center']['lng']]),
                                num_iters=args['iters'])

    with open(f'{args["output"]}{k}.csv', 'w', newline='') as csvfile:
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
plt.show()
