import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np
import cv2
import os

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import pickle

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def drawBetweenKeypoints(pose_img, keypoints, indexes, color, scaleFactor):
    ind0 = indexes[0] - 1
    ind1 = indexes[1] - 1
    
    point1 = (keypoints[ind0][0], keypoints[ind0][1])
    point2 = (keypoints[ind1][0], keypoints[ind1][1])

    thickness = int(5 // scaleFactor)


    cv2.line(pose_img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), color, thickness)


def drawBetweenKeypointsList(pose_img, keypoints, keypointPairsList, colorsList, scaleFactor):
    for ind, keypointPair in enumerate(keypointPairsList):
        drawBetweenKeypoints(pose_img, keypoints, keypointPair, colorsList[ind], scaleFactor)

def drawBetweenSetofKeypointLists(pose_img, keypoints_set, keypointPairsList, colorsList, scaleFactor):
    for keypoints in keypoints_set:
        drawBetweenKeypointsList(pose_img, keypoints, keypointPairsList, colorsList, scaleFactor)


def padImg(img, size, blackBorder=True):
    left, right, top, bottom = 0, 0, 0, 0

    # pad x
    if img.shape[1] < size[1]:
        sidePadding = int((size[1] - img.shape[1]) // 2)
        left = sidePadding
        right = sidePadding

        # pad extra on right if padding needed is an odd number
        if img.shape[1] % 2 == 1:
            right += 1

    # pad y
    if img.shape[0] < size[0]:
        topBottomPadding = int((size[0] - img.shape[0]) // 2)
        top = topBottomPadding
        bottom = topBottomPadding
        
        # pad extra on bottom if padding needed is an odd number
        if img.shape[0] % 2 == 1:
            bottom += 1

    if blackBorder:
        paddedImg = cv2.copyMakeBorder(src=img, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        paddedImg = cv2.copyMakeBorder(src=img, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_REPLICATE)

    return paddedImg

def smartCrop(img, size, center):

    width = img.shape[1]
    height = img.shape[0]
    xSize = size[1]
    ySize = size[0]
    xCenter = center[0]
    yCenter = center[1]

    if img.shape[0] > size[0] or img.shape[1] > size[1]:


        leftMargin = xCenter - xSize//2
        rightMargin = xCenter + xSize//2
        upMargin = yCenter - ySize//2
        downMargin = yCenter + ySize//2


        if(leftMargin < 0):
            xCenter += (-leftMargin)
        if(rightMargin > width):
            xCenter -= (rightMargin - width)

        if(upMargin < 0):
            yCenter -= -upMargin
        if(downMargin > height):
            yCenter -= (downMargin - height)


        img = cv2.getRectSubPix(img, size, (xCenter, yCenter))



    return img



def calculateScaleFactor(img, size, poseSpanX, poseSpanY):

    poseSpanX = max(poseSpanX, size[0])

    scaleFactorX = 1


    if poseSpanX > size[0]:
        scaleFactorX = size[0] / poseSpanX

    scaleFactorY = 1
    if poseSpanY > size[1]:
        scaleFactorY = size[1] / poseSpanY

    scaleFactor = min(scaleFactorX, scaleFactorY)


    return scaleFactor



def scaleImg(img, size, poseSpanX, poseSpanY, scaleFactor):
    scaledImg = img

    scaledImg = cv2.resize(img, (0, 0), fx=scaleFactor, fy=scaleFactor) 

    return scaledImg, scaleFactor


def find_keypoints(img_path, pose_estimator):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path)

    return pose_results[0].pred_instances.keypoints  #json.dumps()


# creates and resizes a pose estimate image
def create_animal_pose_image(original_img, workdir='/root', resize=True):
    
    pose_config = f'{workdir}/mmpose/configs/animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py' # path to the model's configuration file
    pose_checkpoint = './models/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth' # path to the model's checkpoint file

    # run inference on the image using the model
    device = 'cuda:0'
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=False)))


    # build pose estimator
    pose_estimator = init_pose_estimator(
        pose_config,
        pose_checkpoint,
        device=device,
        cfg_options=cfg_options
    )


    keypoints = find_keypoints(original_img, pose_estimator)[0]


    # don't use keypoints that go outside the frame in calculations for the center
    interorKeypoints = keypoints[((keypoints[:,0] > 0) & (keypoints[:,0] < original_img.shape[1])) & ((keypoints[:,1] > 0) & (keypoints[:,1] < original_img.shape[0]))]

    xVals = interorKeypoints[:,0]
    yVals = interorKeypoints[:,1]

    minX = np.amin(xVals)
    minY = np.amin(yVals)
    maxX = np.amax(xVals)
    maxY = np.amax(yVals)

    poseSpanX = maxX - minX
    poseSpanY = maxY - minY

    # find mean center

    xSum = np.sum(xVals)
    ySum = np.sum(yVals)

    xCenter = xSum // xVals.shape[0]
    yCenter = ySum // yVals.shape[0]
    center_of_keypoints = (xCenter,yCenter)

    pose_img = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype = np.uint8)

    # order of the keypoints for AP10k and a standardized list of colors for limbs
    keypointPairsList = [(1,2), (2,3), (1,3), (3,4), (4,9), (9,10), (10,11), (4,6), (6,7), (7,8), (4,5), (5,15), (15,16), (16,17), (5,12), (12,13), (13,14)]
    colorsList = [(255,255,255), (100,255,100), (150,255,255), (100,50,255), (50,150,200), (0,255,255), (0,150,0), (0,0,255), (0,0,150), (255,50,255), (255,0,255), (255,0,0), (150,0,0), (255,255,100), (0,150,0), (255,255,0), (150,150,150)] # 16 colors needed

    size = (512,512)
    scaleFactor = calculateScaleFactor(pose_img, size, poseSpanX, poseSpanY)
    drawBetweenKeypointsList(pose_img, keypoints, keypointPairsList, colorsList, scaleFactor)


    pose_img = padImg(pose_img, size)
    pose_img, scaleFactor = scaleImg(pose_img, size, poseSpanX, poseSpanY, scaleFactor)
    if resize:
        rescaledCenter = (center_of_keypoints[0]*scaleFactor, center_of_keypoints[1]*scaleFactor)
        pose_img = smartCrop(pose_img, size, rescaledCenter)

    return pose_img