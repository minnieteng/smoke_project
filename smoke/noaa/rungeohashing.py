import sys, traceback
import os
from geohashparser import GeohashParser
import logging
import time


# source_shape_file_path = "C:/temp/2018-test/"
# results_geojson_file_path = "c:/temp/geojson/" # make sure the folders are created
# results_file_path = "c:/temp/npy/"


def main():
    sys.path.append('C:/Users/melxt/anaconda3/Lib/site-packages')
    start = time.time()
    if len(sys.argv)<=1:
        print("rungeohashing <source_shape_file_path> <results_file_path> <log_file_name>")
    source_shape_file_path = "C:/temp/2019-test/" if len(sys.argv)<=1 else sys.argv[1]
    results_file_path = "c:/temp/10km_grids/" if len(sys.argv)<=2 else sys.argv[2]
    log_file_name = "c:/temp/log/geohashparser10kmgrids2019_0730.log" if len(sys.argv)<=3 else sys.argv[3]
    print("args:" + source_shape_file_path+" ,"+results_file_path+", "+log_file_name)
    hours_of_day_to_exclude = {10,11,12,13,14,15,16,17,18} #hours to exclude in UTC
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file_name)
    filecount = 0
    for root,dirs,files in os.walk(source_shape_file_path):
        for file in files:
            with open(os.path.join(root,file),"r"):
                if file.endswith(".shp"):
                    filecount += 1
                    try:
                        filename = file.replace(".shp","")
                        g = GeohashParser(source_shape_file_path+filename+"/", file, results_file_path, hours_of_day_to_exclude)
                        g.parse()
                        break
                    except:
                        traceback.print_exc(file=sys.stdout)
                        logger.error('failed to parse file:'+source_shape_file_path+filename+"/"+file)
                        continue
    end = time.time()
    print("total files parsed:" + str(filecount) + ", elapsed time:" + str(round(end-start,1)))


if __name__ =="__main__":
    main()
    
    