import cv2
from cv2 import dnn_superres


# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('./test_images/image4.jpg')

scale_percent = 60

#calculate the 50 percent of original dimensions
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
# dsize
dsize = (width, height)
image = cv2.resize(image, dsize)
# Read the desired model
path = "model/EDSR_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite("./resized_images/upscaled4.jpg", result)