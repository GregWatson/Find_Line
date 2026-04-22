# core routines used for line finding

def find_start_point(gray):
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            if gray[y, x] > 0:
                return (x, y)
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