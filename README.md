Author: Shivpratap Singh Panwar {Kody Technolab Public Limited}
Date of creation : 22 July 2026 

Intented cause: 
The project's aim is to full fill the requirement of data tools for Computer Vision task, which requires a variety of tools to create a custom datasets.

Current Supports: 
Annotation:
1. Pascal VOC- Plain BBOX stored in xml folmate ( xmin,ymin,xmax,ymax) [dragNcreate.py]
2. Coco json - json formate ( bbox+sementaion+other coco types) [Polygon_Advanced_Annotation_Tool.py,"takes four or more points"]
3. sam2.1 tiny for coco segmentation complete + OpenCV, Complete mask cordinates with unlimited points. 
Cleaning: 
1. All kinds of cleaning you can imagin(duplucation, similarity,name_renaming,train_test_split cleaning etc)