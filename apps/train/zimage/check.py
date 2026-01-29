import csv 
from glob import glob

file = "captions.csv"
reader = csv.DictReader(open(file))

import os
images = [os.path.basename(image) for image in glob("datasets/hisoka/*")]

image_set = set(images)
csv_set = set()
for row in reader:
    image_path = os.path.basename(row["image_path"])
    csv_set.add(image_path)
    

print(len(image_set), len(csv_set))
print(image_set - csv_set)
