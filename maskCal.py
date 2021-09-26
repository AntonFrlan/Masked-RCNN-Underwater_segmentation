import os
import cv2
import json
import numpy as np
from myInterpreter import uBroj

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

source_folder = os.path.join(os.getcwd(), "images")
json_path = "annotations.json"  # Relative to root directory
count = 0  # Count of total images saved
file_bbs = {}  # Dictionary containing polygon coordinates for mask
file_mask = {}
file_names = {}
current_image = {}
MASK_WIDTH = 1080  # Dimensions should match those of ground truth image
MASK_HEIGHT = 1920

# Read JSON file
with open(json_path) as f:
    data = json.load(f)


def add_to_dict(data, itr, key, count):
    try:
        x_points = data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
        region_attribute = data[itr]["regions"][count]["region_attributes"]["Type"]
        fname = data[itr]["filename"]
    except:
        print("No BB. Skipping", key)
        return

    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])

    file_bbs[key] = all_points
    file_mask[key] = region_attribute
    file_names[key] = fname


for itr in data:
    file_name_json = data[itr]["filename"]
    sub_count = 0  # Contains count of masks for a single ground truth image

    if len(data[itr]["regions"]) > 1:
        for _ in range(len(data[itr]["regions"])):
            key = file_name_json[:-4] + "*" + str(sub_count + 1)
            add_to_dict(data, itr, key, sub_count)
            sub_count += 1
    else:
        add_to_dict(data, itr, file_name_json[:-4], 0)

print("\nDict size: ", len(file_bbs))

image_folder = os.path.join(source_folder, "images")
mask_folder = os.path.join(source_folder, "masks")
annotation_folder = os.path.join(source_folder, "annotations")


def spremi(ime, mask):
    cv2.imwrite(os.path.join(mask_folder, ime + "_mask.png"), mask)


pointOff = [[0, 10], [5, 5], [10, 0], [5, -5], [0, -10], [-5, -5], [-10, 0], [-5, 5], [3, 7], [-3, 7], [3, -7], [-3, 7],
            [15, 7], [-15, 7], [15, -7], [-15, 7], [15, 11], [-15, 11], [15, -11], [-15, 11]]
def inPoly(point, poli):
    polygon = Polygon(poli)
    for i in pointOff:
        newPoint = point + i
        if 1920 > newPoint[0] >= 0 and 1080 > newPoint[1] >= 0:
            testPoint = Point(newPoint)
            if polygon.contains(testPoint):
                return newPoint
    return None


ime = None
mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
key = None
num_masks = None
for itr in file_bbs:
    num_masks = itr.split("*")
    key = itr
    boja2 = 0

    if ime == None:
        ime = file_names[itr].split(".")[0]
        boja = 0
        annote = open(os.path.join(annotation_folder, ime + ".txt"), "w")
    elif ime != file_names[itr].split(".")[0]:
        spremi(ime, mask)
        annote.close()
        count += 1
        mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
        print()
        print("promjena", ime, file_names[itr].split(".")[0])
        ime = file_names[itr].split(".")[0]
        boja = 0
        annote = open(os.path.join(annotation_folder, ime + ".txt"), "w")

    try:
        arr = np.array(file_bbs[itr])
        vrsta = file_mask[itr]
        if vrsta == "Optezivac":
            boja2 = 160
        elif vrsta == "Cijev":
            boja2 = 255
        elif vrsta == "More":
            boja2 = 70

    except:
        print("Not found:", itr)
        continue

    point = inPoly(arr[2], arr)
    if point is None:
        print("Ujeba si => " + ime)
    elif mask[point[1], point[0]] == 0:
        print(ime)
        boja += 1
        vrsta = file_mask[itr]
        annote.write(str(uBroj(vrsta)) + " ")
        cv2.fillPoly(mask, [arr], color=(boja))

spremi(ime, mask)
count += 1

print("Images saved:", count)
