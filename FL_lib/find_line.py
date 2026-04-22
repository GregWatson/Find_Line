import numpy as np
from FL_lib.fl_core import find_start_point, get_adjacent_points

GREY = 127

def find_line(start_point, next_point, gray):
    x1, y1 = start_point
    x2, y2 = next_point
    gray[y1, x1] = GREY
    gray[y2, x2] = GREY

    # get angle of the line
    sum_of_angles = np.arctan2(y2 - y1, x2 - x1)
    points = [next_point]
    print(f"Angle of the line to {next_point}:", np.degrees(sum_of_angles), "degrees")

    while True:
        last_point = next_point 
        next_point = None
        adjacent_points = get_adjacent_points(gray, last_point)
        line_length = np.hypot(last_point[0] - x1, last_point[1] - y1)
        avg_angle = sum_of_angles / len(points)
        if len(adjacent_points) == 0:
            print(f"No more adjacent points found. Start {start_point} -> {last_point} len {line_length}  Avg angle: {np.degrees(avg_angle)} degrees")
            return


        if len(adjacent_points) == 1:
            next_point = adjacent_points[0]
            best_angle = np.arctan2(next_point[1] - start_point[1], next_point[0] - start_point[0])
            print(f"Only one adjacent point found.")
        else:
            # find the adjacent point that is closest to the average angle so far.
            best_angle_diff = float('inf')
            for point in adjacent_points:
                angle = np.arctan2(point[1] - start_point[1], point[0] - start_point[0])
                angle_diff = abs(angle - avg_angle)
                if angle_diff < best_angle_diff:
                    best_angle_diff = angle_diff
                    next_point = point
                    best_angle = angle
            print(f"Multiple adjacent points found.")

        points.append(next_point)
        sum_of_angles += best_angle
        diff_angle = abs(best_angle - avg_angle)
        tol = np.radians(45/line_length)  # tolerance decreases as line gets longer
        print(f"Continuing to {next_point} at angle {np.degrees(best_angle)} degrees  Diff = {np.degrees(diff_angle)} degrees  Tol={np.degrees(tol)}  Line length: {line_length}")

        if diff_angle > tol:
            print(f"Angle difference {np.degrees(diff_angle)} exceeds tolerance. Stopping line.")
            return

        # clear point or we will find it again
        for adjacent_point in adjacent_points:
            gray[adjacent_point[1], adjacent_point[0]] = GREY
        
    return