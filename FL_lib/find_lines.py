import cv2
import numpy as np
import sys
from FL_lib.fl_core import find_initial_start_point, find_local_start_point, get_adjacent_points
from FL_lib.find_line import find_line

# Finds lines within the given BGR image and returns a list of lines, 
# where each line is represented as a tuple of (points, angle). 
# Points is a list of (x, y) coordinates along the line, 
# and angle is the average angle of the line in radians. 
# The function also returns the grayscale version of the input image 
# with the lines marked in grey. 
# The len_thresh parameter specifies the minimum length of a line to be considered valid. 
# Lines shorter than this threshold will be discarded. 

BLACK = 0

def find_lines(bgr_img, len_thresh=10, debug=False):
    lines_found = []
    # Convert the image to simple binary format (grayscale)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    #find a starting point for the line
    start_point = find_initial_start_point(gray)
    if debug: print(f"Initial start point: {start_point}")

    while start_point:
        # # get list of points adjacent to starting point
        # adjacent_points = get_adjacent_points(gray, start_point)
        # if len(adjacent_points) == 0:
        #     if debug:
        #         print("ERROR: No adjacent points found for the starting point.")
        #     return lines_found, gray

        # if debug:
        #     print("Starting point:", start_point)
        #     print("Adjacent points:", adjacent_points)

        points, angle = find_line(start_point, gray, len_thresh=len_thresh, debug=debug)

        gray[start_point[1], start_point[0]] = BLACK  # remove the starting point from the image
        last_point = start_point
        if len(points)>1:
            last_point = points[-1]
            lines_found.append((points, angle))
            if debug: print(f"Added line to lines_found: {points}, {angle:.2f} ")
            # remove used points from the image
            # for point in points:
            #     gray[point[1], point[0]] = 80

        # find the next start point somewhere close to the last point.
        start_point = find_local_start_point(gray, last_point, size_thresh=10, color_thresh=127)
        if start_point: 
            if debug: print(f"Found next start point: {start_point} which was close to last point {last_point}.") 
        if not start_point:
            if debug:
                print("Couldn't find a new start point near last end point - searching entire image.")
            start_point = find_initial_start_point(gray)
        else: 
            if debug:
                print("Found next new start point near last end point:", start_point)
    return lines_found, gray