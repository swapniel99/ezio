import matplotlib.pyplot as plt
import math

def draw_graph(num_iterations, lr_min, lr_max, step_size):
  ix, iy = [],[]
  for i in range(num_iterations):
    ix.append(i)
    cycle = math.floor(1 + (i/(2 * step_size)))
    x = abs((i/step_size) - (2 * cycle)+1)
    lr = lr_min + (lr_max-lr_min) * (1 - x)
    iy.append(lr)
  plt.plot(ix, iy)
  plt.show()