import cv2
import numpy as np
from FL_lib.find_lines import find_lines

# Draw square spiral
def run_test_3(test_params):
    test_num = 3
    print(f"Running test {test_num}...")
    SIZE = test_params['CANVAS_SIZE']
    # create a white canvas of size SIZExSIZE
    img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)
    WHITE = (255,255,255)

    x=1; y=1; line_len = SIZE-4
    lines_drawn = 0
    exp_lines=[]
    while line_len > 11:
        # draw line from (x,y) to (x+line_len,y)
        cv2.line(img, (x,y), (x+line_len,y), WHITE, thickness=1)
        exp_lines.append(((x,y),(x+line_len,y+1)))
        cv2.line(img, (x+line_len,y), (x+line_len,y+line_len), WHITE, thickness=1)
        exp_lines.append(((x+line_len,y+2),(x+line_len-1,y+line_len)))
        cv2.line(img, (x+line_len,y+line_len), (x,y+line_len), WHITE, thickness=1)
        exp_lines.append(((x+line_len-2,y+line_len),(x,y+line_len-1)))
        cv2.line(img, (x, y+line_len), (x,y+2), WHITE, thickness=1)
        exp_lines.append(((x,y+line_len-2),(x+1,y+2)))
        cv2.line(img, (x, y+2), (x+2,y+2), WHITE, thickness=1)
        lines_drawn += 4
        x=x+2; y=y+2; line_len -= 4
    lines_drawn -=1  # last line ends up too short

    if test_params['debug']:
        scaled_img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Initial", scaled_img)

    lines_found, gray = find_lines(img, test_params['LEN_THRESH'], debug=test_params['debug'])

    if test_params['debug']:
        scaled_img = cv2.resize(gray, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Done", scaled_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for i, line in enumerate(lines_found):
        points, angle = line
        e_ends = exp_lines[i]
        if (points[0] != e_ends[0] or points[-1] != e_ends[-1]) and (points[0] != e_ends[-1] or points[-1] != e_ends[0]):
             print(f"*** FAIL ***: Test {test_num} failed: Expected line from {e_ends[0]} to {e_ends[-1]}, found from {points[0]} to {points[-1]}")
             return False
        # print(f"Exp line {e_ends[0]} - {e_ends[-1]}, Found line {points[0]} - {points[-1]}")

    if len(lines_found) != lines_drawn:
        print(f"Test {test_num} failed: Expected {lines_drawn} lines, found {len(lines_found)}")
        return False
    points, _ = lines_found[0]

    return True


