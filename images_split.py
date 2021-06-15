import os
import random
import shutil

# For Images
images = os.listdir("Data/training_Images")
print(len(images))
testing_set = random.sample(images, 1620)

for image in images:
    if image in testing_set:
        shutil.move("Data/training_Images/"+image, "Data/testing_images/"+image)
