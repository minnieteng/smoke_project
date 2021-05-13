import math
from box.Box import Box
import numpy as np

class BoxMapper:
    """ maps lat/lon from geohashparser to Box
    """
    nw_lat, nw_lon = 57.870760, -133.540154
    sw_lat_est, sw_lon_est = 46.173395, -129.055971
    dist = 1250
    res = 10
    #query_lat, query_lon = 53.913068, -122.827957 
    box = None
    hours_of_day_to_exclude = []
    hourly_boxes = [[]] * 24
    #cell_index = Box.get_cell_assignment_if_in_grid(box,query_lat,query_lon)
    
    def __init__(self, hours_of_day_to_exclude):
        self.box = Box(self.nw_lat, self.nw_lon, self.sw_lat_est, self.sw_lon_est, self.dist, self.res)
        self.hours_of_day_to_exclude = hours_of_day_to_exclude
        for hour in range(0,24):
            if hour in self.hours_of_day_to_exclude:
                self.hourly_boxes[hour] = np.full((Box.get_num_cells(self.box), Box.get_num_cells(self.box)),-1)
            else:
                self.hourly_boxes[hour] = np.full((Box.get_num_cells(self.box), Box.get_num_cells(self.box)),0)
                
    def append(self,hour_of_day,lat,lon,feature):
        cell_coords = Box.get_cell_assignment_if_in_grid(self.box,lat,lon)
        if math.isnan(cell_coords[0]) or math.isnan(cell_coords[1]):
            return
        x, y = int(cell_coords[0]), int(cell_coords[1])
        hour_box = self.hourly_boxes[hour_of_day]
        if hour_box[x][y] != 1:
            hour_box[x][y] = feature
    
    def to_array(self, hour_of_day):
        return self.hourly_boxes[hour_of_day]

# append(self,time,lat,lon,feature):
    # coord = getcoord(lat,lon)
    # if boxes[time] is None:
        # boxes[time] = np.empty((Box.Box.get_num_cells(box), Box.Box.get_num_cells(box)))
    # if boxes[time] [coord.x][coord.y] is not 1:
        # boxes[time] [coord.x][coord.y] = feature

 