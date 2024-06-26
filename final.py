# Import necessary libraries
#%%
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from skimage.util import random_noise, img_as_float
from time import perf_counter
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter  # Importing image-related modules from the Python Imaging Library (PIL)
import multiprocessing as mp  # Importing the multiprocessing module

# Function to convert an image to black and white
def convert_image_BW(name):
    # Open the image file
    imgBW = Image.open(f"Images/{name}") 
    # Convert the image to grayscale
    imgBW = imgBW.convert("L") 
    # Convert the grayscale image to a NumPy array
    imgBW = np.array(imgBW)
    # Calculate the median pixel value
    median = np.median(imgBW) 
    # Convert pixels below the median to black and pixels above to white
    imgBW = np.uint8((imgBW < median) * 255) 
    # Convert the NumPy array back to an image
    imgBW = Image.fromarray(imgBW) 
    # Save the black and white image
    imgBW.save(f"blacked/{name}")

# Function to apply blur to an image
def Blur_image(name):
    # Open the image file
    img = Image.open(f"Images/{name}")
    # Apply blur filter to the image
    img = img.filter(ImageFilter.BLUR) 
    # Save the blurred image. Adjust folder name if needed.
    img.save(f"blurred/{name}")

# Function to add noise to a black and white image
def Add_noise_BW(name): 
    # Open the black and white image file
    imgBWN = Image.open(f"blacked/{name}") 
    # Open the original
    imgOriginal = Image.open(f"Images/{name}")
    # Convert the black one to a NumPy array
    imgBWN = np.array(imgBWN) 
    # Calculate the number of noisy pixels to be added
    noisy_pixels = np.sum(imgBWN == 0) / np.size(imgBWN) * 0.1
    # Add salt and pepper noise to the image
    noise = random_noise(img_as_float(imgOriginal), mode='s&p', amount=noisy_pixels)
    # Convert the NumPy array back to an image
    noisy_img = Image.fromarray((noise * 255).astype(np.uint8))
    # Save the noisy image. Adjust folder name if needed
    noisy_img.save(f"noised/{name}")

# Function to process images using multiprocessing
def process_chunk(chunk):
    convert_image_BW(chunk)
    Blur_image(chunk)
    Add_noise_BW(chunk)

# Main function for parallelized image processing
def parallelized_conversion(filenames, n_splits):
    # Print a message indicating the start of image processing
    print("Starting run.")
    # Record the start time
    start_time = perf_counter()
    # Process filenames
    with ProcessPoolExecutor(max_workers=n_splits) as executor:
        executor.map(process_chunk, filenames)
    # Calculate the total processing time
    timing = perf_counter() - start_time
    # Print a message indicating the completion of image processing and the total time taken
    print(f"Run done. Time: {timing} s")

# Entry point of the script
if __name__ == '__main__':
    # Get the list of filenames from the 'Images' directory. Adjust if needed
    filenames = np.array(os.listdir('Images'))
    # Get the maximum number of CPUs available
    cpus_max = mp.cpu_count()
    # Dictionary to store processing times for different numbers of CPUs
    timings = {}

    # Iterate over different numbers of CPUs
    for c in range(1, cpus_max + 1):
        print(f"Using {c} CPUs")
        # Record the start time
        start_time = perf_counter()
        # Perform parallelized image processing with the current number of CPUs
        parallelized_conversion(filenames, c)
        # Calculate the total processing time
        timings[c] = perf_counter() - start_time
    
    # Plot the performance (processing time) vs number of CPUs used
    plt.plot(timings.keys(), timings.values())
    plt.title('Processing time vs number of CPUs')
    plt.ylabel('Processing time (seconds)')
    plt.xlabel('Number of CPUs')
    plt.show() 

# %%
