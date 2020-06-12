# Object-Detection-using-Proposal-generators

## Selective Search 
It is an object detection technique which involves over segmentation using methods like FH segmentation and then compares the similarity between neighboring regions and merges the pair with highest similarity and repeats this process until a defined criteria is reached. Similarity can be mesured in terms color, size, shape and texture.   

#### Algorithm implementation
• The algorithm first reads image using cv2.imread command. The images are read in BGR format
• Images are converted from BGR to RGB format
• Argument parser takes in image path, annotation path and strategy as inputs. However, default paths for image and annotations are mentioned in the code
• cv2.ximgproc.segmentation contains Selective Search functions. First, a Selective Search Segmentation instance is created (ss)
• We create a graph segmentation method and add it to the ss instance
• Now the algorithm checks if the input argument parsed is color or all strategies. If color, an instance is created using createSelectiveSearchSegmentationStrategyColor(), if all, an instance is created using createSelectiveSearchSegmentationStrategyMultiple() and all the strategies color, fill, size and texture are added into the list of the instance
• Add strategy to the ss instance
• ss.process command is used to create boxes with limit set to 100 boxes. This returns x,y, width(w) and height(h)
