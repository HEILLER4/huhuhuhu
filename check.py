import json
with open('C://Users//lbert//PycharmProjects//libra//Vehicles-coco.v2i.coco//train//_annotations.coco.json') as f:
    data = json.load(f)

print("Categories:", data['categories'])
print("First annotation:", data['annotations'][0])