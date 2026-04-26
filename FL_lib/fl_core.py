# core routines used for line finding
import numpy as np

# Calculate angle in radians between the line from start_point to pt and the horizontal axis.
# Always returns a positive angle between 0 and 2*pi.
def get_angle(pt, start_point):
    a = np.arctan2(pt[1] - start_point[1], pt[0] - start_point[0])
    while a < 0.0:
        a += 2*np.pi
    return a

def find_initial_start_point(gray):
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            if gray[y, x] >= 127:
                return (x, y)
    return None

# Search an ever-expanding circle around the last point, looking for a set pixel.
def find_local_start_point(gray, last_point, size_thresh=10, color_thresh=255):
    # print(f"Searching for local start point around {last_point} with size_thresh {size_thresh} and color_thresh {color_thresh}.")
    x, y = last_point
    for dist in range(1, size_thresh + 1):
        for new_y in range(y - dist, y + dist + 1):
            if new_y in [y - dist, y + dist]:
                for new_x in range(x-dist, x+dist+1):
                    if 0 <= new_x < gray.shape[1] and 0 <= new_y < gray.shape[0]:
                        if gray[new_y, new_x] >= color_thresh:
                            return (new_x, new_y)
            else:
                for new_x in [x - dist, x + dist]:
                    if 0 <= new_x < gray.shape[1] and 0 <= new_y < gray.shape[0]:
                        if gray[new_y, new_x] >= color_thresh:
                            return (new_x, new_y)

    return None


def get_adjacent_points(gray, point, thresh=255):
    x, y = point
    adjacent_points = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < gray.shape[1] and 0 <= new_y < gray.shape[0]:
                if gray[new_y, new_x] >= thresh:
                    adjacent_points.append((new_x, new_y))
    return adjacent_points