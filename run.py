import cv2, sys
from animal_pose_tools import create_animal_pose_image

input_file = 'test_imgs/dog.png'
output_file = 'out.png'

photo = cv2.imread(input_file)
pose = create_animal_pose_image(photo)
cv2.imwrite(output_file, pose)