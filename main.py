import cv2
import numpy as np
import argparse
import sys
from FL_lib.find_lines import find_lines
from Tests.tests import run_tests

def main():
    parser = argparse.ArgumentParser(description="Piece Project CLI")
    parser.add_argument("-t", "--tests", action="store_true", help="Run tests")
    args = parser.parse_args()

    if args.tests:
        test_params = { 'CANVAS_SIZE': 20, 
                        'LEN_THRESH':10,
                        'LOG': False }
        passed, total = run_tests(test_params)
        return
    
    SIZE = 20
    # create a white canvas of size SIZExSIZE
    img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)

    # draw a line
    cv2.line(img, (1,1), (18,18), (255, 255, 255), thickness=1)
    cv2.line(img, (1,2), (2,1), (255, 255, 255), thickness=1)

    lines_found, gray = find_lines(img, len_thresh=10, debug=True)
    for line in lines_found:
        points, angle = line
        print(f"Found line {points[0]} - {points[-1]} with angle: {np.degrees(angle):.2f} degrees")

    # SCALE IMAGE TO 500X500
    scaled_img = cv2.resize(gray, (500, 500), interpolation=cv2.INTER_NEAREST)

    # Display the image
    cv2.imshow("Line", scaled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
