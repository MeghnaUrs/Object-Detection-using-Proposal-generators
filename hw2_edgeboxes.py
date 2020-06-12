# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:51:46 2019
@author: Meghana Kiran Urs
USC ID: 1211214571
EE677 hw2_edgeboxes.py 
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET

def get_groundtruth_boxes(annoted_img_path):
    """
    :param annoted_img_path:
    Find root and object/bndbox in the xml file
    Form a list (gt_boxes) containing x and y coordinates of the ground truth boxes
    :return gt_boxes:
    """    
    gt_boxes = []
    tree = ET.parse(annoted_img_path)
    root = tree.getroot()
    for items in root.findall('object/bndbox'):
        xmin = items.find('xmin')
        ymin = items.find('ymin')
        xmax = items.find('xmax')
        ymax = items.find('ymax')
        gt_boxes.append([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
    return gt_boxes


def get_intersection_area(box1, box2):
    """
    :param box1:
    :param box2:
    :return intersection area:
    """
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    if (x2 - x1 < 0) or (y2 - y1 < 0):
        return 0.0
    else:
        return (x2 - x1 + 1) * (y2 - y1 + 1)


def calculate_iou(proposal_boxes, gt_boxes):
    """
    :param proposal_boxes:
    :param gt_boxes:
    :return:
    Calculate area of ground truth boxes
    Calculate area of proposal boxes
    Calculate intersection and union area 
    Calculate IoU
    Find all proposal boxes whose IoU is greater than 0.5
    Find the best IoU box match for each ground truth box
    :return iou_qualified_boxes, final_boxes:
    """
    iou_qualified_boxes = []
    final_boxes = []
    for gt_box in gt_boxes:
        best_box_iou = 0
        best_box = 0
        area_gt_box = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])                       
        for prop_box in proposal_boxes:
            area_prop_box = (prop_box[2] - prop_box[0] + 1) * (prop_box[3] - prop_box[1] + 1)  
            intersection_area = get_intersection_area(prop_box, gt_box)
            union_area = area_prop_box + area_gt_box - intersection_area               
            iou = float(intersection_area) / float(union_area)                       
            if iou > 0.5:
                iou_qualified_boxes.append(prop_box)
                if iou > best_box_iou:
                    best_box_iou = iou
                    best_box = prop_box
        if best_box_iou != 0:
            final_boxes.append(best_box)
    return iou_qualified_boxes, final_boxes


if __name__ == "__main__":

#Define image path, annotation path and strategy. Read the image and convert it to RGB from BGR
#Create a selective segmentation instance ss 
#Create a graph segmentation method

    annotated_img_path = 'C:/Users/12134/OneDrive/Desktop/Computer Vision/HW2_Data/Annotations/002129.xml'
    img = cv2.imread('C:/Users/12134/OneDrive/Desktop/Computer Vision/HW2_Data/JPEGImages/002129.jpg')

    # Download model from https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz
    model = 'C:/Users/12134/OneDrive/Desktop/Computer Vision/HW2_Data/model.yml.gz'

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # get the edges
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
    # create an orientation map
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    #Create edge boxes
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setAlpha(0.7)
    edge_boxes.setBeta(0.6)
    edge_boxes.setMaxBoxes(100)
    prop_boxes = edge_boxes.getBoundingBoxes(edges, orimap)


    # Convert x,y,w,h parameters for the op 100 proposal boxes into
    # x, y, x+w, y+h parameters to be consistent with the xml tags of the ground truth boxes
    boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in prop_boxes]

    output_img_proposal_top100 = img.copy()
    output_img_iou_qualified = img.copy()
    output_img_final = img.copy()

    gt_boxes = get_groundtruth_boxes(annotated_img_path)
    print("Number of Ground Truth Boxes = ", len(gt_boxes))

    # Draw bounding boxes for top 100 proposals
    for i in range(0, len(boxes)):
        top_x, top_y, bottom_x, bottom_y = boxes[i]
        cv2.rectangle(output_img_proposal_top100, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Output_Top_100_Proposals", output_img_proposal_top100)
    cv2.waitKey()
    cv2.destroyAllWindows()

    iou_qualified_boxes, final_boxes = calculate_iou(boxes, gt_boxes)
    print("Number of Qualified Boxes with IOU > 0.5 = ", len(iou_qualified_boxes))
    print("Qualified Boxes = ", iou_qualified_boxes)

    # Draw bounding boxes for iou_qualified_boxes
    for i in range(0, len(iou_qualified_boxes)):
        top_x, top_y, bottom_x, bottom_y = iou_qualified_boxes[i]
        cv2.rectangle(output_img_iou_qualified, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA)
    for i in range(0, len(gt_boxes)):
        top_x, top_y, bottom_x, bottom_y = gt_boxes[i]
        cv2.rectangle(output_img_iou_qualified, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Output_IOU_Qualified_Proposals", output_img_iou_qualified)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("Number of final boxes = ", len(final_boxes))
    print("Final boxes = ", final_boxes)

    recall = len(final_boxes) / len(gt_boxes)
    print("Recall = ", recall)

    # Draw bounding boxes for final_boxes
    for i in range(0, len(final_boxes)):
        top_x, top_y, bottom_x, bottom_y = final_boxes[i]
        cv2.rectangle(output_img_final, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA)
    for i in range(0, len(gt_boxes)):
        top_x, top_y, bottom_x, bottom_y = gt_boxes[i]
        cv2.rectangle(output_img_final, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Output_Final_Boxes", output_img_final)
    cv2.waitKey()
    cv2.destroyAllWindows()
