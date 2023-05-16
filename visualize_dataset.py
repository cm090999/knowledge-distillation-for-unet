import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def normalize_image(image):
    # Normalize image to uint8
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = (image * 255).astype(np.uint8)
    return image

def plot_images(image1, image2):
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(image1)
    axs[0].set_title('Image 1')

    axs[1].imshow(image2)
    axs[1].set_title('Image 2')

    plt.show()

def process_images(folder1, folder2, imgnumber = 0):
    startwithvalue = str(imgnumber).zfill(5)
    for filename in os.listdir(folder1):
        if filename.endswith('.png') and filename.startswith(startwithvalue):
            image_path1 = os.path.join(folder1, filename)
            image_path2 = os.path.join(folder2, filename)

            if os.path.isfile(image_path2):
                image1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
                image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)

                # Normalize images
                image1 = normalize_image(image1)
                image2 = normalize_image(image2)

                # Plot images
                plot_images(image1, image2)
                pass


# Specify the folders containing the images
folder1 = 'dataset/diodeMasks/inputs'
folder2 = 'dataset/diodeMasks/masks'
process_images(folder1, folder2)
for i in range(100):
    process_images(folder1=folder1,
                   folder2=folder2,
                   imgnumber=i)
    
    plt.waitforbuttonpress()
