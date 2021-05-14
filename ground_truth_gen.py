import os
import json
from tqdm import tqdm
from collections import defaultdict

file_path = input("BG Image: ")
sets = {
    "train": "instances_train2017.json",
    "validation": "instances_val2017.json"
}
filename = sets['train']
data = json.load(open(os.path.join("annotations/", filename)))

image_annotations = defaultdict(lambda: None)

for annot in tqdm(data["annotations"], desc="Parsing Annotations"):
    if image_annotations[annot["image_id"]] is None:
        image_annotations[annot["image_id"]] = [(annot["bbox"], annot["category_id"])]
    else:
        image_annotations[annot["image_id"]].append((annot["bbox"], annot["category_id"]))

cate = {x['id']: x['name'] for x in json.load(open(os.path.join("annotations/", sets["validation"])))['categories']}

image_dir = {
    "train": "train2017",
    "validation": "val2017"
}
for im in data['images']:
    if im['file_name'] == file_path:
        im_height = im["height"]
        im_width = im["width"]
        image_id = im["id"]
        if image_annotations[image_id] is not None:
            f = open("ground_truth/" + im['file_name'][:-4] + ".txt", "w+")
            for i in range(len(image_annotations[image_id])):
                bbox = image_annotations[image_id][i][0]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, im_width - 1)
                y2 = min(y2, im_height - 1)
                f.write(cate[image_annotations[image_id][i][1]] + " %d %d %d %d\r\n" % (
                    x1, y1, x2, y2))
        break

