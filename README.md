# Object-Detection-using-Proposal-generators

## Selective Search 
It is an object detection technique which involves over segmentation using methods like FH segmentation and then compares the similarity between neighboring regions and merges the pair with highest similarity and repeats this process until a defined criteria is reached. Similarity can be mesured in terms color, size, shape and texture.   

#### Algorithm implementation:
- The algorithm first reads image using cv2.imread command. The images are read in BGR format
- Images are converted from BGR to RGB format
- Argument parser takes in image path, annotation path and strategy as inputs. However, default paths for image and annotations are mentioned in the code
- cv2.ximgproc.segmentation contains Selective Search functions. First, a Selective Search Segmentation instance is created (ss)
- We create a graph segmentation method and add it to the ss instance
- Now the algorithm checks if the input argument parsed is color or all strategies. If color, an instance is created using createSelectiveSearchSegmentationStrategyColor(), if all, an instance is created using createSelectiveSearchSegmentationStrategyMultiple() and all the strategies color, fill, size and texture are added into the list of the instance
- Add strategy to the ss instance
- ss.process command is used to create boxes with limit set to 100 boxes. This returns x,y, width(w) and height(h)

#### Result:

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/43301609/84458016-e0dd1800-ac18-11ea-81b0-d367a1402f27.png">
</p>


## Edge Boxes
It is a method of locating object proposals using edges. It involves detecting edges, thresholding them, non maximal suppression to remove low scoring boxes.

#### Algorithm implementation:
-	Find x and y coordinates of the bounding boxes for the top 100 proposals, x, y, x+w, y+h. These parameters are consistent with the xml tags provided for ground truths
-	A function is written to find x and y coordinates of the bounding boxes for ground truths using the coordinates given in the annotations xml file using xml.etree.ElementTree module
-	This module finds the root and the tags given by object/bndbox
-	Draw bounding boxes for the top 100 proposals using cv2.rectangle command
-	A function is written to calculate IoU and find the coordinate of the boxes whose IoU is greater than 0.5. This is found by calculating the area of the bounding box of top proposals and area of ground truth boxes
-	Intersection area:

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/43301609/84457884-9fe50380-ac18-11ea-95be-a9b0e1a796ad.PNG">
</p>

This is calculated by finding the minimum and maximum values of x and y coordinates of both the boxes inside a function. This can be visualized in the above figure
-	Union area is nothing but (area of box 1 + area of box 2 – area of intersection). We subtract area of intersection since it is considered twice, once in area of box 1 and again in box 2 area
-	IoU = Intersection area / Union area 
-	Boxes with IoU greater than 0.5 are considered as qualified boxes and one best box of every proposal with IoU > 0.5 is again filtered and stored in a final boxes list
-	This is used to calculate the recall
-	Recall = Number of final boxes / Total ground truth boxes
-	All the qualified boxes and final boxes are separately drawn on output images using cv2.rectangle 

#### Result:

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/43301609/84458049-f7836f00-ac18-11ea-9832-dc26f365c966.png">
</p>
