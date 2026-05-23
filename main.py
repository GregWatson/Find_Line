import cv2
import numpy as np
import argparse
import sys
from FL_lib.find_lines import find_lines
from Tests.tests import run_tests
from FL_lib.pre_proc_image import pre_process_image
from FL_lib.polygon_angles import get_polygon_angles
from FL_lib.find_corners import find_corners

def main():
    parser = argparse.ArgumentParser(description="Piece Project CLI")
    parser.add_argument("-t", "--tests", action="store_true", help="Run tests")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode with verbose output")
    args = parser.parse_args()

    if args.tests:
        test_params = { 'CANVAS_SIZE': 20, 
                        'LEN_THRESH':10,
                        'LOG': False,
                        'debug': args.debug }
        passed, total = run_tests(test_params)
        return
    
    SIZE = 500
    # create a white canvas of size SIZExSIZE
    img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)

    # Define vertices of the polygon (must be int32)
    poly_points = np.array([[100, 100], [300, 100], [250, 150], [250, 300], [100, 300]], np.int32)

    cv2.fillPoly(img, [poly_points], color=(255, 255, 255)) 

    cv2.imshow("Line", img)

    # get image with just edges
    pre_processed_image = pre_process_image(img, debug=args.debug)
    edges = cv2.Canny(pre_processed_image, 50, 150, apertureSize=3)

    cv2.imshow("Edges", edges)

    # cv2.line(img, (1,18), (18,1), (255, 255, 255), thickness=1)

    lines_found = find_lines(edges, len_thresh=10, debug=args.debug)
    if not len(lines_found):
        print("No lines were found.")
    else:
        print(f"Found {len(lines_found)} lines:")
        lines_found_img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)
        for line in lines_found:
            points, angle = line
            cv2.line(lines_found_img, points[0], points[-1], (255, 255, 255), thickness=1)
            print(f"Found line {points[0]} - {points[-1]} with angle: {np.degrees(angle):.1f} degrees")

    angles = get_polygon_angles(poly_points)
    print("Polygon angles (inner, outer):")
    for i, (inner, outer) in enumerate(angles):
        print(f"Vertex {i} {poly_points[i]}: Inner angle = {inner:.2f} degrees, Outer angle = {outer:.2f} degrees")

    just_lines = [(line[0][0], line[0][-1])  for line in lines_found]
    corners = find_corners(just_lines, debug=args.debug)
    print(f"Found {len(corners)} corners:")
    for i, (cornerness, point, angle) in enumerate(corners):
        print(f"Corner {i}: cornerness={cornerness:.2f}, point={point}, angle={np.degrees(angle):.2f} degrees")
        cv2.circle(lines_found_img, point, 5, (0, 0, 255), -1)
    cv2.imshow("Lines Found", lines_found_img)
    
    # Display the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
