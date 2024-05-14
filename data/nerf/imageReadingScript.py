import os
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test3/images/train/", "0001.png")
# Open the PNG image file
image = Image.open(image_path)

# Convert the image to RGBA mode (if it's not already in RGBA mode)
image = image.convert("RGBA")

# Get the pixel data
pixel_data = image.load()

# Get the width and height of the image
width, height = image.size

# Iterate over each pixel in the image
for y in range(height):
    for x in range(width):
        # Get the RGBA values for the pixel
        r, g, b, a = pixel_data[x, y]

        # Do something with the alpha value
        print(f"Alpha value for pixel at ({x}, {y}): {a}")
