'''
Perform a Hough transform procedure to localize a lat/lon coordinate
to a particular Box grid assignment in a robust fashion
'''
from __future__ import division

import math
import os
import sys
import time

import geopy as gp
import numpy as np
from geopy.distance import distance
from scipy.optimize import Bounds, minimize

from collections import Counter


def three_nw(x, nw_dist, ne_dist, se_dist, dist):
    curr_nw_dist = math.sqrt(x[0] ** 2 + x[1] ** 2)
    curr_ne_dist = math.sqrt((dist - x[0]) ** 2 + x[1] ** 2)
    curr_se_dist = math.sqrt((dist - x[0]) ** 2 + (dist - x[1]) ** 2)

    return (curr_nw_dist - nw_dist) ** 2 + (curr_ne_dist - ne_dist) ** 2 + (curr_se_dist - se_dist) ** 2

def three_ne(x, ne_dist, se_dist, sw_dist, dist):
    curr_ne_dist = math.sqrt((dist - x[0]) ** 2 + x[1] ** 2)
    curr_se_dist = math.sqrt((dist - x[0]) ** 2 + (dist - x[1]) ** 2)
    curr_sw_dist = math.sqrt(x[0] ** 2 + (dist - x[1]) ** 2)

    return (curr_sw_dist - sw_dist) ** 2 + (curr_ne_dist - ne_dist) ** 2 + (curr_se_dist - se_dist) ** 2

def three_se(x, nw_dist, sw_dist, se_dist, dist):
    curr_nw_dist = math.sqrt(x[0] ** 2 + x[1] ** 2)
    curr_sw_dist = math.sqrt(x[0] ** 2 + (dist - x[1]) ** 2)
    curr_se_dist = math.sqrt((dist - x[0]) ** 2 + (dist - x[1]) ** 2)

    return (curr_nw_dist - nw_dist) ** 2 + (curr_sw_dist - sw_dist) ** 2 + (curr_se_dist - se_dist) ** 2

def three_sw(x, nw_dist, ne_dist, sw_dist, dist):
    curr_nw_dist = math.sqrt(x[0] ** 2 + x[1] ** 2)
    curr_ne_dist = math.sqrt((dist - x[0]) ** 2 + x[1] ** 2)
    curr_sw_dist = math.sqrt(x[0] ** 2 + (dist - x[1]) ** 2)

    return (curr_nw_dist - nw_dist) ** 2 + (curr_ne_dist - ne_dist) ** 2 + (curr_sw_dist - sw_dist) ** 2

def four_optim(x, nw_dist, ne_dist, se_dist, sw_dist, dist):
    curr_nw_dist = math.sqrt(x[0] ** 2 + x[1] ** 2)
    curr_ne_dist = math.sqrt((dist - x[0]) ** 2 + x[1] ** 2)
    curr_se_dist = math.sqrt((dist - x[0]) ** 2 + (dist - x[1]) ** 2)
    curr_sw_dist = math.sqrt(x[0] ** 2 + (dist - x[1]) ** 2)

    return ((curr_nw_dist - nw_dist) ** 2 + (curr_ne_dist - ne_dist) ** 2 + 
            (curr_se_dist - se_dist) ** 2 + (curr_sw_dist - sw_dist) ** 2)

# Given Euclidean distance, return cell assignment
def get_cell_assignment(x, y, dist, num_cells):
    return int(np.floor(x / dist * num_cells)), int(np.floor(y / dist * num_cells))

if __name__ == "__main__":
    # Set the box corners here for now (later they will be class variables)
    nw_lat, nw_lon = 56.956768, -131.38922
    sw_lat, sw_lon = 47.30657, -126.70448
    ne_lat, ne_lon = 58.88212191, -112.77439365
    se_lat, se_lon = 48.84712669, -111.84507173

    # Again, dist and resolution will be a class variables
    dist = 1120
    resolution = 5
    num_cells = dist // resolution

    # Given a query point:
    query_lat, query_lon = 53.913068, -122.827957   # Somewhere near Prince George

    # Compute distance to each corner
    nw_dist = distance((nw_lat, nw_lon), (query_lat, query_lon)).km
    sw_dist = distance((sw_lat, sw_lon), (query_lat, query_lon)).km
    ne_dist = distance((ne_lat, ne_lon), (query_lat, query_lon)).km
    se_dist = distance((se_lat, se_lon), (query_lat, query_lon)).km

    # Get a Euclidean assignment from each pair of corner optimizations
    init_x = np.ones((2,)) * dist / 2
    bounds = Bounds([0, 0], 
                    [dist, dist])
    cell_assignments = []

    # Get nw_optim
    tic = time.time()
    res = minimize(three_nw, init_x,
                    args=(nw_dist, ne_dist, se_dist, dist),
                    method='SLSQP', tol=1e-6, bounds=bounds)
    x, y = res.x[0], res.x[1]
    cell_x, cell_y = get_cell_assignment(x, y, dist, num_cells)
    cell_assignments.append((cell_x, cell_y))
    toc = time.time()
    print(res)
    print('Time elapsed: %.7f' % (toc - tic))

    # Get ne_optim
    tic = time.time()
    res = minimize(three_ne, init_x,
                    args=(ne_dist, se_dist, sw_dist, dist),
                    method='SLSQP', tol=1e-6, bounds=bounds)
    x, y = res.x[0], res.x[1]
    cell_x, cell_y = get_cell_assignment(x, y, dist, num_cells)
    cell_assignments.append((cell_x, cell_y))
    toc = time.time()
    print(res)
    print('Time elapsed: %.7f' % (toc - tic))

    # Get se_optim
    tic = time.time()
    res = minimize(three_se, init_x,
                    args=(nw_dist, sw_dist, se_dist, dist),
                    method='SLSQP', tol=1e-6, bounds=bounds)
    x, y = res.x[0], res.x[1]
    cell_x, cell_y = get_cell_assignment(x, y, dist, num_cells)
    cell_assignments.append((cell_x, cell_y))
    toc = time.time()
    print(res)
    print('Time elapsed: %.7f' % (toc - tic))

    # Get sw_optim
    tic = time.time()
    res = minimize(three_sw, init_x,
                    args=(nw_dist, ne_dist, sw_dist, dist),
                    method='SLSQP', tol=1e-6, bounds=bounds)
    x, y = res.x[0], res.x[1]
    cell_x, cell_y = get_cell_assignment(x, y, dist, num_cells)
    cell_assignments.append((cell_x, cell_y))
    toc = time.time()
    print(res)
    print('Time elapsed: %.7f' % (toc - tic))

    # Get four_optim
    tic = time.time()
    res = minimize(four_optim, init_x,
                    args=(nw_dist, ne_dist, se_dist, sw_dist, dist),
                    method='SLSQP', tol=1e-6, bounds=bounds)
    x, y = res.x[0], res.x[1]
    cell_x, cell_y = get_cell_assignment(x, y, dist, num_cells)
    cell_assignments.append((cell_x, cell_y))
    toc = time.time()
    print(res)
    print('Time elapsed: %.7f' % (toc - tic))

    # Take mode of cell assignments (only if 3 or more agreements, otherwise, throw an error here)
    mode_counter = Counter(cell_assignments)
    [(mode, _)] = mode_counter.most_common(1)
    print(mode)
    print(mode_counter[mode])
    assert(mode_counter[mode] >= 3)
