from __future__ import division

import math
import os
import sys
import time

import geopy as gp
import numpy as np
# from geopy.distance import great_circle as distance
from geopy.distance import distance
from scipy.optimize import Bounds, minimize

from Box import Box

def frac_lat_lon(box, nw_square, sw_square, ne_square, se_square):
        # Compute box area w/ Heron's formula
        def heron(a, b, c):
            dist_1 = distance(a, b).km
            dist_2 = distance(a, c).km
            dist_3 = distance(c, b).km

            s = (dist_1 + dist_2 + dist_3) / 2

            return math.sqrt(s * (s - dist_1) * (s - dist_2) * (s - dist_3))

        top_left_square = heron(nw_square, sw_square, ne_square)
        bot_right_square = heron(se_square, sw_square, ne_square)
        square_area = top_left_square + bot_right_square

        top_left_box = heron((box.nw_lat, box.nw_lon), (box.sw_lat, box.sw_lon), (box.ne_lat, box.ne_lon))
        bot_right_box = heron((box.se_lat, box.se_lon), (box.sw_lat, box.sw_lon), (box.ne_lat, box.ne_lon))
        box_area = top_left_box + bot_right_box

        print('Num lat/lon we expect inside a centered box: %.4f' % (box_area / square_area * 144 * 288))

if __name__ == "__main__":
    # # Top left and bottom left of the square
    # nw_lat, nw_lon = 56.956768, -131.38922
    # sw_lat_est, sw_lon_est = 48.541751, -129.580869
    # dist = 1120
    # res = 5
    
    nw_lat, nw_lon = 57.870760, -133.540154
    sw_lat_est, sw_lon_est = 46.173395, -129.055971
    dist = 1250
    res = 5

    box = Box(nw_lat, nw_lon, sw_lat_est, sw_lon_est, dist, res)

    # m, n = box.get_cell_assignment(stat_lat, stat_lon)
    # print(m, n)
    # sys.exit(1)


    # print(box.nw_lat)
    # print(box.nw_lon)
    # print()

    # print(box.sw_lat)
    # print(box.sw_lon)
    # print()

    # print(box.ne_lat)
    # print(box.ne_lon)
    # print()

    # print(box.se_lat)
    # print(box.se_lon)
    # print()

    # northernmost_lat, nothernmost_lon = 56.24472, -120.85611
    # print(box.is_within(northernmost_lat, nothernmost_lon))
    # sys.exit(1)

    # all_lat = np.linspace(46.95, 61.95, 144)
    # all_lon = np.linspace(-140.0, -114, 288)
    # all_lat = np.linspace(46.95454545454545, 61.95454545454545, 166)
    # all_lon = np.linspace(-139.9547038327526, -113.95470383275261, 288)
    # nw_square = (61.95454545454545, -139.9547038327526)
    # sw_square = (46.95454545454545, -139.9547038327526)
    # ne_square = (61.95454545454545, -113.95470383275261)
    # se_square = (46.95454545454545, -113.95470383275261)
    # query_lat, query_lon = 49.317698, -117.661527
    # print(box.is_within(query_lat, query_lon))


    # Compute cell assignment for Victoria station
    query_lat, query_lon = 48.58, -123.4422

    # Init
    # Compute distance to each corner
    nw_dist = distance((box.nw_lat, box.nw_lon), (query_lat, query_lon)).km
    sw_dist = distance((box.sw_lat, box.sw_lon), (query_lat, query_lon)).km
    ne_dist = distance((box.ne_lat, box.ne_lon), (query_lat, query_lon)).km
    se_dist = distance((box.se_lat, box.se_lon), (query_lat, query_lon)).km

    init_x = np.ones((2,)) * dist / 2
    bounds = Bounds([0, 0], 
                    [dist, dist])

    # # Test corner computations
    # res = minimize(box._three_nw, init_x,
    #                     args=(nw_dist, ne_dist, se_dist),
    #                     method='SLSQP', tol=1e-6, bounds=bounds)
    # print(res)
    # sys.exit(1)

    # res = minimize(box._three_ne, init_x,
    #                     args=(ne_dist, se_dist, sw_dist),
    #                     method='SLSQP', tol=1e-6, bounds=bounds)
    # print(res)

    # res = minimize(box._three_se, init_x,
    #                     args=(nw_dist, sw_dist, se_dist),
    #                     method='SLSQP', tol=1e-6, bounds=bounds)
    # print(res)

    # res = minimize(box._three_sw, init_x,
    #                     args=(nw_dist, ne_dist, sw_dist),
    #                     method='SLSQP', tol=1e-6, bounds=bounds)
    # print(res)

    # res = minimize(box._four_optim, init_x,
    #                     args=(nw_dist, ne_dist, se_dist, sw_dist),
    #                     method='SLSQP', tol=1e-6, bounds=bounds)
    # print(res)


    # print()
    # print()
    # box.get_cell_assignment(query_lat, query_lon)

    box.visualize_box()
    sys.exit(1)

    # frac_lat_lon(box, nw_square, sw_square, ne_square, se_square)

    # tic = time.time()
    # out_of_box_count = 0
    # for lat in all_lat:
    #     for lon in all_lon:
    #         if not box.is_within(lat, lon):
    #             continue
    #         p, q = box.get_cell_assignment(lat, lon)
    #         if np.isnan(p):
    #             out_of_box_count += 1
    # toc = time.time()
    # print('Time elapsed: %.7f' % (toc - tic))
    # print(box.err_count)
    # print(box.correct_count)
    # print(box.total_count)
    # print(out_of_box_count)
