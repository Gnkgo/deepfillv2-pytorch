import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Sample JSON data
json_data = """
{
    "10_0": [[184, 254, 251, 303], [-34, 252, 214, 483]], 
    "10_1": [[123, 188, 160, 211], [99, 375, -15, 229]], 
    "10_2": [[142, 206, 124, 164], [-71, 184, -1, 214]], 
    "10_3": [[198, 270, 226, 280], [209, 512, 121, 321]]
}
"""

# Parse the JSON data
data = json.loads(json_data)

# Load the existing image
image = cv2.imread('augmented_images/10_0.png')

# Initialize a mask with the same dimensions as the image
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Get the rectangles for "10_0"
# Get the shapes for "10_0"
shapes_to_plot = data.get("10_0", [])

# Draw shapes on the mask
for shape in shapes_to_plot:
    pts = np.array(shape).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)  # Fill the region with white
    


# Overlay the mask on the image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display the original image, the mask, and the masked image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
plt.title('Masked Image')
plt.axis('off')

plt.show()
