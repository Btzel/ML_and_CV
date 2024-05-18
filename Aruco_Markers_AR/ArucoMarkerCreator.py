# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 03:00:04 2024

@author: Burak Tüzel
"""

# import the necessary packages
import numpy as np
import argparse
import cv2
import sys


# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def generate_aruco_marker(output_path, marker_id, marker_type):
    # Define the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[marker_type])

    # Allocate memory for the output ArUco tag
    tag = np.zeros((300, 300, 1), dtype="uint8")

    # Draw the ArUco tag on the output image
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, 300, tag, 1)

    # Write the generated ArUco tag to disk
    cv2.imwrite(output_path, tag)

    # Display the ArUco tag (optional)
    cv2.imshow("ArUco Tag", tag)
    cv2.waitKey(0)

if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(description="Generate ArUco marker.")
    parser.add_argument("--output", default="tags/DICT_5x5_100_id70.png", help="Output file path for the ArUco marker")
    parser.add_argument("--id", type=int, default=70, help="ID of the ArUco marker")
    parser.add_argument("--type", default="DICT_5X5_100", choices=ARUCO_DICT.keys(), help="Type of the ArUco marker")

    args = parser.parse_args()

    # Verify that the supplied ArUco tag exists and is supported by OpenCV
    if ARUCO_DICT.get(args.type, None) is None:
        print("[INFO] ArUco tag of '{}' is not supported".format(args.type))
        sys.exit(0)

    # Generate the ArUco marker
    generate_aruco_marker(args.output, args.id, args.type)