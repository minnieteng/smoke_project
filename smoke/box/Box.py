from __future__ import division

import math
import os
import sys
import time
from collections import Counter

import geopy as gp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import pandas as pd
from geopy.distance import distance
from mpl_toolkits.basemap import Basemap
from scipy.optimize import Bounds, minimize
from shapely.geometry import Point, Polygon

import logging

class Box():
    def __init__(self, nw_lat, nw_lon, sw_lat_est, sw_lon_est, dist, res):
        # Save exact args used to make Box
        self.orig_box_args =  (nw_lat, nw_lon, sw_lat_est, sw_lon_est, dist, res)

        # Create dictionary to store previously assigned coordinates
        self.previous_assignments = {}

        self.nw_lat = nw_lat
        self.nw_lon = nw_lon
        self.sw_lat = sw_lat_est
        self.sw_lon = sw_lon_est
        self.ne_lat = None
        self.ne_lon = None
        self.se_lat = None
        self.se_lon = None

        # # FOR DEBUGGING PURPOSES:
        # self.err_count = 0
        # self.correct_count = 0
        # self.total_count = 0

        self.dist = dist
        self.res = res
        self.num_cells = dist // res

        # Calculate index of last cell
        self.last_cell_indx = self.num_cells - 1

        self._compute_box_coords()

        self.poly = Polygon([(self.nw_lat, self.nw_lon), (self.ne_lat, self.ne_lon),
                    (self.se_lat, self.se_lon), (self.sw_lat, self.sw_lon)])

        # Get logger
        self.logger = logging.getLogger(__name__)

    def _assert_optim(self, res, optim_func):
        try:
            assert(res.message == 'Optimization terminated successfully.')
        except AssertionError:
            print(res)
            print('%s optimization call did not terminate successfully for %.5f, %.5f' % (optim_func.__name__))
            sys.exit(1)

    def _assert_assign_optim(self, res, optim_func, lat, lon):
        try:
            assert(res.message == 'Optimization terminated successfully.')
        except AssertionError:
            # print(res)
            # print('%s optimization call did not terminate successfully for %.5f, %.5f' % (optim_func.__name__, lat, lon))
            raise

    def _find_sw_corner(self, x):
        nw_dist = distance((self.nw_lat, self.nw_lon), (x[0], x[1])).km

        return (nw_dist - self.dist) ** 2

    def _find_ne_corner(self, x):
        nw_dist = distance((self.nw_lat, self.nw_lon), (x[0], x[1])).km
        sw_dist = distance((self.sw_lat, self.sw_lon), (x[0], x[1])).km

        return (nw_dist - self.dist) ** 2 + (sw_dist - self.dist * math.sqrt(2)) ** 2

    def _find_se_corner(self, x):
        nw_dist = distance((self.nw_lat, self.nw_lon), (x[0], x[1])).km
        sw_dist = distance((self.sw_lat, self.sw_lon), (x[0], x[1])).km
        ne_dist = distance((self.ne_lat, self.ne_lon), (x[0], x[1])).km

        return (nw_dist - self.dist * math.sqrt(2)) ** 2 + (sw_dist - self.dist) ** 2 + (ne_dist - self.dist) ** 2

    def _compute_box_coords(self):
        # # 1. Solve for a correction to sw_lat, sw_lon such that dist is satisfied
        # tic = time.time()
        bounds = Bounds([self.sw_lat - 5, self.sw_lon + 1],
                        [self.sw_lat + 5, self.sw_lon + 5])
        res = minimize(self._find_sw_corner, [self.sw_lat, self.sw_lon],
                        method='SLSQP', tol=1e-6, bounds=bounds)
        # toc = time.time()
        # print('Time elapsed: %.7f' % (toc - tic))
        # print(res)
        self._assert_optim(res, self._find_sw_corner)
        self.sw_lat, self.sw_lon = res.x[0], res.x[1]
        # print('%.5f, %.5f' % (sw_lat, sw_lon))
        # print()
        # print('Vertical distance %.3f' % distance((self.sw_lat, self.sw_lon), (self.nw_lat, self.nw_lon)).km)
        # print()

        # # 2. Now we define a search problem for another vertex of the square (NE)
        # We can use nw_lat to define the initial guess for one coordinate of the x axis
        self.ne_lat, self.ne_lon = self.nw_lat, self.nw_lon

        # To localize NE corner of square, we define the bounds as follows:
        bounds = Bounds([self.sw_lat + 3, self.ne_lon],
                    [self.ne_lat + 10, self.ne_lon + 50])
        # tic = time.time()
        res = minimize(self._find_ne_corner, [self.ne_lat, self.ne_lon],
                        method='SLSQP', tol=1e-6, bounds=bounds, options={'maxiter': 100})
        self._assert_optim(res, self._find_ne_corner)
        self.ne_lat, self.ne_lon = res.x[0], res.x[1]
        # toc = time.time()
        # print(res)
        # print('Time elapsed: %.7f' % (toc - tic))

        # print('Horizontal distance %.3f' % distance((self.nw_lat, self.nw_lon), (self.ne_lat, self.ne_lon)).km)
        # print('Diagonal distance %.3f' % distance((self.sw_lat, self.sw_lon), (self.ne_lat, self.ne_lon)).km)
        # print()

        # # 3. Now to compute the last vertex (SE)
        self.se_lat, self.se_lon = self.sw_lat, self.ne_lon

        # To localize SE corner of square, we use the following bounds:
        bounds = Bounds([self.se_lat - 10, self.ne_lon - 10],
                        [self.se_lat + 10, self.ne_lon + 10])
        # tic = time.time()
        res = minimize(self._find_se_corner, [self.se_lat, self.se_lon],
                        method='SLSQP', tol=1e-6, bounds=bounds)
        self._assert_optim(res, self._find_se_corner)
        self.se_lat, self.se_lon = res.x[0], res.x[1]
        # toc = time.time()
        # print(res)
        # print('Time elapsed: %.7f' % (toc - tic))

        # print('Horizontal distance %.3f' % distance((self.sw_lat, self.sw_lon), (self.se_lat, self.se_lon)).km)
        # print('Diagonal distance %.3f' % distance((self.nw_lat, self.nw_lon), (self.se_lat, self.se_lon)).km)
        # print('Vertical distance %.3f' % distance((self.ne_lat, self.ne_lon), (self.se_lat, self.se_lon)).km)

    def _three_nw(self, x, nw_dist, ne_dist, se_dist):
        curr_nw_dist = math.sqrt(x[0] ** 2 + x[1] ** 2)
        curr_ne_dist = math.sqrt((self.dist - x[0]) ** 2 + x[1] ** 2)
        curr_se_dist = math.sqrt((self.dist - x[0]) ** 2 + (self.dist - x[1]) ** 2)

        return (curr_nw_dist - nw_dist) ** 2 + (curr_ne_dist - ne_dist) ** 2 + (curr_se_dist - se_dist) ** 2

    def _three_ne(self, x, ne_dist, se_dist, sw_dist):
        curr_ne_dist = math.sqrt((self.dist - x[0]) ** 2 + x[1] ** 2)
        curr_se_dist = math.sqrt((self.dist - x[0]) ** 2 + (self.dist - x[1]) ** 2)
        curr_sw_dist = math.sqrt(x[0] ** 2 + (self.dist - x[1]) ** 2)

        return (curr_sw_dist - sw_dist) ** 2 + (curr_ne_dist - ne_dist) ** 2 + (curr_se_dist - se_dist) ** 2

    def _three_se(self, x, nw_dist, sw_dist, se_dist):
        curr_nw_dist = math.sqrt(x[0] ** 2 + x[1] ** 2)
        curr_sw_dist = math.sqrt(x[0] ** 2 + (self.dist - x[1]) ** 2)
        curr_se_dist = math.sqrt((self.dist - x[0]) ** 2 + (self.dist - x[1]) ** 2)

        return (curr_nw_dist - nw_dist) ** 2 + (curr_sw_dist - sw_dist) ** 2 + (curr_se_dist - se_dist) ** 2

    def _three_sw(self, x, nw_dist, ne_dist, sw_dist):
        curr_nw_dist = math.sqrt(x[0] ** 2 + x[1] ** 2)
        curr_ne_dist = math.sqrt((self.dist - x[0]) ** 2 + x[1] ** 2)
        curr_sw_dist = math.sqrt(x[0] ** 2 + (self.dist - x[1]) ** 2)

        return (curr_nw_dist - nw_dist) ** 2 + (curr_ne_dist - ne_dist) ** 2 + (curr_sw_dist - sw_dist) ** 2

    def _four_optim(self, x, nw_dist, ne_dist, se_dist, sw_dist):
        curr_nw_dist = math.sqrt(x[0] ** 2 + x[1] ** 2)
        curr_ne_dist = math.sqrt((self.dist - x[0]) ** 2 + x[1] ** 2)
        curr_se_dist = math.sqrt((self.dist - x[0]) ** 2 + (self.dist - x[1]) ** 2)
        curr_sw_dist = math.sqrt(x[0] ** 2 + (self.dist - x[1]) ** 2)

        return ((curr_nw_dist - nw_dist) ** 2 + (curr_ne_dist - ne_dist) ** 2 +
                (curr_se_dist - se_dist) ** 2 + (curr_sw_dist - sw_dist) ** 2)

    # Given Euclidean distance, return cell assignment
    def _get_euclidean_assignment(self, x, y):
        return int(np.floor(x / self.dist * self.last_cell_indx)), int(np.floor(y / self.dist * self.last_cell_indx))

    def get_cell_assignment(self, query_lat, query_lon):
        # Compute distance to each corner
        nw_dist = distance((self.nw_lat, self.nw_lon), (query_lat, query_lon)).km
        sw_dist = distance((self.sw_lat, self.sw_lon), (query_lat, query_lon)).km
        ne_dist = distance((self.ne_lat, self.ne_lon), (query_lat, query_lon)).km
        se_dist = distance((self.se_lat, self.se_lon), (query_lat, query_lon)).km

        # Get a Euclidean assignment from each pair of corner optimizations
        init_x = np.ones((2,)) * self.dist / 2
        bounds = Bounds([0, 0],
                        [self.dist, self.dist])
        cell_assignments = []

        # Get nw_optim
        res = minimize(self._three_nw, init_x,
                        args=(nw_dist, ne_dist, se_dist),
                        method='SLSQP', tol=1e-6, bounds=bounds)
        x, y = res.x[1], res.x[0]
        cell_x, cell_y = self._get_euclidean_assignment(x, y)
        try:
            self._assert_assign_optim(res, self._three_nw, query_lat, query_lon)
        except AssertionError:
            cell_x, cell_y = np.nan, np.nan
        cell_assignments.append((cell_x, cell_y))

        # Get ne_optim
        res = minimize(self._three_ne, init_x,
                        args=(ne_dist, se_dist, sw_dist),
                        method='SLSQP', tol=1e-6, bounds=bounds)
        x, y = res.x[1], res.x[0]
        cell_x, cell_y = self._get_euclidean_assignment(x, y)
        try:
            self._assert_assign_optim(res, self._three_ne, query_lat, query_lon)
        except AssertionError:
            cell_x, cell_y = np.nan, np.nan
        cell_assignments.append((cell_x, cell_y))

        # Get se_optim
        # tic = time.time()
        res = minimize(self._three_se, init_x,
                        args=(nw_dist, sw_dist, se_dist),
                        method='SLSQP', tol=1e-6, bounds=bounds)
        x, y = res.x[1], res.x[0]
        cell_x, cell_y = self._get_euclidean_assignment(x, y)
        try:
            self._assert_assign_optim(res, self._three_se, query_lat, query_lon)
        except AssertionError:
            cell_x, cell_y = np.nan, np.nan
        cell_assignments.append((cell_x, cell_y))

        # Get sw_optim
        res = minimize(self._three_sw, init_x,
                        args=(nw_dist, ne_dist, sw_dist),
                        method='SLSQP', tol=1e-6, bounds=bounds)
        x, y = res.x[1], res.x[0]
        cell_x, cell_y = self._get_euclidean_assignment(x, y)
        try:
            self._assert_assign_optim(res, self._three_sw, query_lat, query_lon)
        except AssertionError:
            cell_x, cell_y = np.nan, np.nan
        cell_assignments.append((cell_x, cell_y))

        # Get four_optim
        res = minimize(self._four_optim, init_x,
                        args=(nw_dist, ne_dist, se_dist, sw_dist),
                        method='SLSQP', tol=1e-6, bounds=bounds)
        x, y = res.x[1], res.x[0]
        cell_x, cell_y = self._get_euclidean_assignment(x, y)
        try:
            self._assert_assign_optim(res, self._four_optim, query_lat, query_lon)
        except AssertionError:
            cell_x, cell_y = np.nan, np.nan
        cell_assignments.append((cell_x, cell_y))

        # Take mode of cell assignments (only if 3 or more agreements, otherwise, throw an error here)
        mode_counter = Counter(cell_assignments)
        [(mode, _)] = mode_counter.most_common(1)
        # print(mode)
        # print(mode_counter[mode])
        # self.total_count += 1
        try:
            assert(mode_counter[mode] >= 3)
            # if mode_counter[mode] == mode_counter[(cell_x, cell_y)]:
            #     self.err_count += 1
            # self.correct_count += 1
        except AssertionError:
            self.logger.debug('No agreement of at least 3 computations for % .5f, %.5f' % (query_lat, query_lon))
            mode = (cell_x, cell_y)
            # print(mode_counter)
            # sys.exit(1)

        return mode

    def is_within(self, query_lat, query_lon):
        p = Point(query_lat, query_lon)

        return self.poly.contains(p)

    def visualize_box(self):
        # Make the background map
        # m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=80)  # full map
        m = Basemap(llcrnrlon=-140, llcrnrlat=45, urcrnrlon=-100, urcrnrlat=70)
        m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
        m.fillcontinents(color='grey', alpha=0.3)
        m.drawcoastlines(linewidth=0.1, color="white")

        # Add a point per position
        lat = [self.nw_lat, self.ne_lat, self.se_lat, self.sw_lat, self.nw_lat]
        lon = [self.nw_lon, self.ne_lon, self.se_lon, self.sw_lon, self.nw_lon]

        x, y = m(lon, lat)
        m.plot(x, y, 'o-', markersize=5, linewidth=1)
        # lons = [self.nw_lat, self.ne_lat]
        # m.scatter(stations_df_relevant['LONG'], stations_df_relevant['LAT'],
        #             alpha=0.4, cmap="Set1")

        plt.savefig('box.png')

    def get_num_cells(self):
        """ Return the number of cells in the theoretical box

        :returns: Number of cells in box on either axis
        :rtype: int
        """
        return self.num_cells

    def get_orig_box_args(self):
        """ Return a tuple of the original arguments used to make the box
        in order nw_lat, nw_lon, sw_lat_est, sw_lon_est, dist, res

        :returns: Original args used to make Box
        :rtype: tuple
        """
        return self.orig_box_args

    def is_already_assigned(self, query_lat, query_lon):
        """ Returns true if tuple of lat, lon have already been assigned

        :param query_lat: Latitude to check
        :type query_lat: float
        :param query_lon: Longitude to check
        :type query_lon: float
        :returns: Boolean of whether pair was assigned already or not
        :rtype: bool
        """
        return (query_lat, query_lon) in self.previous_assignments.keys()

    def store_assignment(self, lat_stor, long_stor, row_stor, col_stor):
        """ Stores row, col assignment of paired lat, lon to be used if lat, lon
        pair appear again

        :param lat_stor: Latitude to store assignment of
        :type lat_stor: float
        :param long_stor: Longitude to store assignment of
        :type long_stor: float
        :param row_stor: Row assignment to store
        :type row_stor: float
        :param col_stor: Column assignment to store
        :type col_stor: float
        """
        self.previous_assignments[(lat_stor, long_stor)] = (row_stor, col_stor)

    def get_previous_assignment(self, stored_lat, stored_lon):
        """ Returns previous assignment tuple of row, col for given
        stored_lat, stored_lon coordinates, assumes given coords were
        previously assigned and stored

        :param lat_stor: Latitude to get assignment of
        :type lat_stor: float
        :param long_stor: Longitude to get assignment of
        :type long_stor: float
        :returns: Previously assigned tuple of row and col for lat and lon
        :rtype: tuple of (float, float)
        """
        return self.previous_assignments.get((stored_lat, stored_lon))

    def get_cell_assignment_if_in_grid(self, query_lat, query_lon):
        """ Assigns latitude and longitude a cell coordinate if they are in
        the grid. Uses previous cell assignment if already assigned before.

        :param query_lat: Latitude to assign cell if coords in grid
        :type query_lat: float
        :param query_lon: Longitude to assign cell if coords in grid
        :type query_lon: float
        :return: Cell assignment tuple (jth row, ith column) if in grid and
                 (np.nan, np.nan) if not within grid
        :rtype: tuple<int or np.nan>
        """
        if not self.is_already_assigned(query_lat, query_lon):

            if self.is_within(query_lat, query_lon):
                row, col = self.get_cell_assignment(query_lat, query_lon)
                self.store_assignment(query_lat, query_lon, row, col)
                return row, col

            else:
                self.store_assignment(query_lat, query_lon, np.nan, np.nan)
                return (np.nan, np.nan)

        else:
            return self.get_previous_assignment(query_lat, query_lon)
