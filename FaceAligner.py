# import the necessary packages
from helpers import FACIAL_LANDMARKS_IDXS
from helpers import shape_to_np
import numpy as np
import cv2
 
class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35)):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        #self.landmarks = landmarks
        self.desiredLeftEye = desiredLeftEye
        #self.desiredFaceWidth = desiredFaceWidth
        #self.desiredFaceHeight = desiredFaceHeight
 

    def align(self, image, landmarks, desiredFaceWidth=112, desiredFaceHeight=None):
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if desiredFaceHeight is None:
            desiredFaceHeight = desiredFaceWidth

        # convert the landmark (x, y)-coordinates to a NumPy array
        #shape = self.predictor(gray, rect)
        shape = landmarks
        shape = np.asarray(shape)
        #shape = shape_to_np(self.landmarks[0])
        
 
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        #leftEyePts = np.asarray(shape[0]["left_eye"])
        #rightEyePts = np.asarray(shape[0]["right_eye"])



        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
 
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180




        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist





        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])



        # apply the affine transformation
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(np.asarray(image), M, (w, h), flags=cv2.INTER_CUBIC)
 
        # return the aligned face
        return output



