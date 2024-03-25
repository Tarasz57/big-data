# Import necessary libraries
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
    imgBW.save(f"Bw/{name}")

# Function to apply blur to an image
def Blur_image(name):
    # Open the image file
    img = Image.open(f"Images/{name}")
    # Apply blur filter to the image
    img = img.filter(ImageFilter.BLUR) 
    # Save the blurred image. Adjust folder name if needed.
    img.save(f"Blur/{name}")

# Function to add noise to a black and white image
def Add_noise_BW(name): 
    # Open the black and white image file
    imgBWN = Image.open(f"Bw/{name}") 
    # Convert the image to a NumPy array
    imgBWN = np.array(imgBWN) 
    # Count the number of black pixels
    black_pixels = np.sum(imgBWN == 0)
    # Calculate the number of noisy pixels to be added
    noisy_pixels = int(black_pixels * 0.1) 
    # Get the coordinates of black pixels
    black_pixels_coords = np.argwhere(imgBWN == 0)
    # Shuffle the coordinates randomly
    np.random.shuffle(black_pixels_coords)
    # Select a subset of coordinates for adding noise
    noisy_coords = black_pixels_coords[:noisy_pixels]
    # Add noise to the selected coordinates
    for coord in noisy_coords:
        imgBWN[coord[0], coord[1]] = 255 
    # Convert the NumPy array back to an image
    noisy_img = Image.fromarray(imgBWN)
    # Save the noisy image. Adjust folder name if needed
    noisy_img.save(f"Bwn/{name}")

# Function to process a chunk of images using multiprocessing
def process_chunk(chunk, task_functions):
    # Process each image in the chunk using multiprocessing
    with ProcessPoolExecutor() as executor:
        for task_function in task_functions:
            executor.map(task_function, chunk)

# Main function for parallelized image processing
def parallelized_conversion(filenames, n_splits, n_cpus):
    # Print a message indicating the start of image processing
    print("Starting run.")
    # Record the start time
    start_time = perf_counter()
    # Split the list of filenames into chunks
    chunks = np.array_split(filenames, n_splits)
    # Process each chunk of filenames
    for chunk in chunks:
        # Process the chunk using multiprocessing
        process_chunk(chunk, [convert_image_BW, Blur_image, Add_noise_BW])
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
        # Record the start time
        start_time = perf_counter()
        # Perform parallelized image processing with the current number of CPUs
        parallelized_conversion(filenames, c, c)
        # Calculate the total processing time
        timings[c] = perf_counter() - start_time
    
    # Plot the performance (processing time) vs number of CPUs used
    plt.plot(timings.keys(), timings.values())
    plt.title('Processing time vs number of CPUs')
    plt.ylabel('Processing time (seconds)')
    plt.xlabel('Number of CPUs')
    plt.show() 
