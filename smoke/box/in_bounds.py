from shapely.geometry import MultiPoint, Point, Polygon

from time import time

from Box import Box

if __name__ == "__main__":
    # Top left and bottom left of the square
    nw_lat, nw_lon = 56.956768, -131.38922
    sw_lat_est, sw_lon_est = 48.541751, -129.580869
    dist = 1120
    res = 5

    box = Box(nw_lat, nw_lon, sw_lat_est, sw_lon_est, dist, res)

    query_lat, query_lon = 53.913068, -122.827957   # Somewhere near Prince George

    tic = time()
    # Define the polygon object from the corners of the square
    poly = Polygon([(box.nw_lat, box.nw_lon), (box.ne_lat, box.ne_lon),
                    (box.sw_lat, box.sw_lon), (box.se_lat, box.se_lon)])
    p = Point(query_lat, query_lon)
    print(poly.contains(p))
    toc = time()
    print('Time elapsed: %.7f' % (toc - tic))
