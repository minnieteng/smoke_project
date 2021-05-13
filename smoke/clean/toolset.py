import numpy as np
import numpy.ma as ma
from abc import ABC, abstractmethod


class CellCruncher(ABC):
    def __init__(self):
        pass

    def crunch_data(self, cell_assignments, data_arr):
        """ Perform aggregation operation on data which are in the same cell and
        remove those without cell assignments

        :param cell_assignments: Cell assignments for each value in data_arr
                                 corresponding to same shape
        :type cell_assignments: np.ma.array
        :param data_arr: Data for which to aggregate similar cells and remove no assigns
        :type data_arr: np.array
        :return: unique cell assignments  and corresponding array of data array
                 with similar cells aggregated (don't need ma.array since all non assigns dropped)
        :rtype: (np.array, np.array)
        """
        # Flatten arrays into 1D of data and coord pairs
        flat_data = data_arr.flatten()
        flat_assignments = cell_assignments.flatten().reshape(flat_data.size, 2)

        # Rebuild mask array to similar shape as flat_assignments if loses shape
        # from ma.array bug
        if type(flat_assignments.mask) == np.bool_ and not flat_assignments.mask:
            flat_assignments = ma.array(flat_assignments.data, mask=False)

        # Filter out any which have an invalid assignment (are masked)
        valid_pairs = np.logical_not(
            flat_assignments.mask.any(-1)
        )  # logical not as mask if False if valid and True if invalid
        n_valid_entries = np.sum(valid_pairs)
        valid_assigns = np.dstack((valid_pairs, valid_pairs))[
            0
        ]  # Remove extra size 1 axis created by 3rd dim concat
        filtered_data = flat_data[valid_pairs]
        filtered_assignments = flat_assignments[valid_assigns].reshape(
            n_valid_entries, 2
        ).data  # No more masked assigns so just go back to normal array

        unique_data = []
        unique_assignments = []

        # Guard against all non selects
        if not (filtered_assignments.size == 0) and not (filtered_data.size == 0):

            # Find all assignment pairs that only occur once (don't need crunching)
            unique_pairs, n_occurences = np.unique(filtered_assignments, axis=0, return_counts=True)
            single_occurence_pairs = unique_pairs[n_occurences == 1]
            non_single_occurence_pairs = unique_pairs[n_occurences > 1]

            # Turn assignment pairs to complex row+col*j for both array of all assignment pairs and
            # array of assignment pairs that happen only once, and use numpy.1d to get
            # bool indices for assignment pairs in filtered_assignments occur only once.
            # COMPLEX NUMBER USED TO TURN 2D pair arr INTO 1D arr for np.in1d use, NO OTHER REASON THEN EFFICIENCY!
            indices_occuring_once = np.in1d(filtered_assignments[:,0]+filtered_assignments[:,1]*1j,
                                            single_occurence_pairs[:,0]+single_occurence_pairs[:,1]*1j)

            # Add all single occurences to unique_data, and unique_assignments as they are since require
            # not crunching
            unique_data += list(filtered_data[indices_occuring_once])
            unique_assignments += list(filtered_assignments[indices_occuring_once])

            # For all assignment pairs that don't only occur once, crunch data to get only unique assignments
            for non_single_occ_assign_pair in non_single_occurence_pairs:
                select_indices = (filtered_assignments == non_single_occ_assign_pair).all(-1)
                overlapping_data = filtered_data[select_indices]
                unique_data.append(self.crunch_similar_cells(overlapping_data))
                unique_assignments.append(non_single_occ_assign_pair)


        return (np.array(unique_assignments), np.array(unique_data))

    @abstractmethod
    def crunch_similar_cells(self, data_arr):
        """ Abstract method to get a resulting single value given a data_arr of overlapping values

        :param data_arr: Overlapping data in a cell
        :type: np.array
        :return: Single value derived from data for cell
        :rtype: float
        """
        ...


class SumCellCruncher(CellCruncher):
    def crunch_similar_cells(self, data_arr):
        """ Crunches data by summing all overlapping data in cell return np.nan
        if all are nan to ensure that no data slices remain no data

        :param data_arr: Overlapping data in a cell
        :type: np.array
        :return: Sum of data for cell
        :rtype: float
        """
        if not np.isnan(data_arr).all():
            return np.nansum(data_arr)
        else:
            return np.nan


class MeanCellCruncher(CellCruncher):
    def crunch_similar_cells(self, data_arr):
        """ Crunches data by averaging all overlapping data in cell

        :param data_arr: Overlapping data in a cell
        :type: np.array
        :return: Mean of data for cell
        :rtype: float
        """
        return np.nanmean(data_arr)


class TimeCruncher(ABC):

    def __init__(self):
        pass

    def crunch_to_result_TTSG(self, time_bin_size_h, original_time_space_grid, result_time_space_grid):
        """ Take the original_time_space_grid and for all space grid's in each time bin,
        specified by result_time_space_grid and time_bin_size, average those space grids
        to get a singular space grid at each time in result_time_space_grid. Return
        result_time_space_grid populated with those crunched space grids.

        # TODO DOCSTRINGS

        """
        # Extract original times, and target times to crunch things to
        orig_times = original_time_space_grid.get_times()
        target_times = result_time_space_grid.get_times()

        # Don't attempt to crunch if no times given for original
        if not (orig_times.size == 0):
            # For every target time, find all space grids in the that time bin, and crunch them down
            # to single grid, then populating the result TTSG with that single grid
            for time_bin_end in target_times:
                relevant_time_indices = ((time_bin_end-np.timedelta64(time_bin_size_h, 'h') < orig_times) &
                                         (orig_times <= time_bin_end))
                space_grids_in_time_bin = original_time_space_grid.get_grid()[relevant_time_indices]
                crunched_grid = self.crunch_to_single_grid(space_grids_in_time_bin)
                result_time_space_grid.set_time_grid(time_bin_end, crunched_grid)

        return result_time_space_grid

    @abstractmethod
    def crunch_to_single_grid(time_space_grid):
        ...


class AvgTimeBinTimeCruncher(TimeCruncher):

    def crunch_to_single_grid(self, time_space_grid):
        """ Take the 3D time space grid of (times_in_time_bin, row, col)
        and average all spaces in time bin to get one single grid

        # TODO DOCSTRINGS

        """
        return np.nanmean(time_space_grid, axis=0)
