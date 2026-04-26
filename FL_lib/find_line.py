import numpy as np
from FL_lib.fl_core import get_adjacent_points, get_angle
from FL_lib.find_min_len_line import find_min_len_line

BLACK = 0
WHITE = 255

# Given a starting point, find a line by following adjacent points in the grayscale image 
# and checking how straight the line is. 
# We use a recursive backtracking approach to try different paths until we find a valid line 
# or exhaust all possibilities.
# On success, returns the list of points along the line and the average angle of the line 
# in radians.

def find_line(start_point, gray, len_thresh=10, debug=False):

    # first, see if we can find a line that is len_thresh or more pixels long,
    # starting from the start point and following adjacent points while checking angle deviation to ensure we 
    # are following a line and not a curve. 
    points, init_avg_angle = find_min_len_line(start_point, [start_point], gray, len_thresh=len_thresh, debug=debug)
    if len(points) < 2: # No valid line
        if debug:
            print(f"No valid line found starting from {start_point}.")
        return [], None

    gray[start_point[1], start_point[0]] = BLACK  # mark the starting point as used in the image
    sum_of_angles = init_avg_angle * len(points)  # we will keep a running sum of angles to calculate the average as we go along

    while True:
        last_point = points[-1]
        x1, y1 = last_point
        next_point = None
        adjacent_points = get_adjacent_points(gray, last_point)
        line_length = np.hypot(last_point[0] - x1, last_point[1] - y1)
        avg_angle = sum_of_angles / len(points)
        if len(adjacent_points) == 0:
            if debug:
                print(f"No more adjacent points found. Start {start_point} -> {last_point} len {line_length}  Avg angle: {np.degrees(avg_angle)} degrees")
            return [start_point] + points, avg_angle

        # clear adajacent points from the image so we don't reuse them
        for pt in adjacent_points:
            gray[pt[1], pt[0]] = BLACK

        if len(adjacent_points) == 1:
            next_point = adjacent_points[0]
            best_angle = get_angle(next_point, start_point)
        else:
            # find the adjacent point that is closest to the average angle so far.
            best_angle_diff = float('inf')
            for point in adjacent_points:
                angle = get_angle(point, start_point)
                angle_diff = abs(angle - avg_angle)
                if angle_diff > 2*np.pi:
                    angle_diff -= 2*np.pi
                if angle_diff < best_angle_diff:
                    best_angle_diff = angle_diff
                    next_point = point
                    best_angle = angle

        points.append(next_point)
        sum_of_angles += best_angle
        diff_angle = abs(best_angle - avg_angle)
        if diff_angle > 2*np.pi:
            diff_angle -= 2*np.pi
        tol = np.radians(90/(line_length+1))  # tolerance decreases as line gets longer

        if diff_angle > tol:
            if debug:
                print(f"Angle difference {np.degrees(diff_angle)} exceeds tolerance. Stopping line.")
            # restore adjacent points that were marked as used but not part of the line
            for pt in adjacent_points:
                if pt != next_point:
                    gray[pt[1], pt[0]] = WHITE
            return points, avg_angle

        if debug:
            print(f"Continuing to {next_point} at angle {np.degrees(best_angle):.2f} degrees  Diff = {np.degrees(diff_angle):.2f} degrees  Tol={np.degrees(tol):.3f}  Line length: {line_length:.1f}")

        # Keep going - try next adjacent point
