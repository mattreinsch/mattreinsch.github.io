import os
import matplotlib.pyplot as plt
from PIL import Image
import random

def display_example_images(directory, num_images=5):
    # List all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith(('jpg', 'jpeg', 'png'))]
    
    # Check if the directory has enough images
    if len(image_files) < num_images:
        print(f"Only {len(image_files)} images found. Displaying all of them.")
        num_images = len(image_files)
    
    # Select a random subset of images
    example_images = random.sample(image_files, num_images)
    
    # Create a subplot grid
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    # Display each image
    for ax, image_file in zip(axes, example_images):
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(image_file)
    
    plt.show()

display_example_images('/kaggle/input/vehicle-classification/Vehicles/Cars', num_images=5)
display_example_images('/kaggle/input/vehicle-classification/Vehicles/Planes', num_images=5)
display_example_images('/kaggle/input/vehicle-classification/Vehicles/Ships', num_images=5)