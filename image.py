from PIL import Image, ImageFilter 
from skimage.util import random_noise, img_as_float
import numpy as np
import multiprocessing as mp
from os import listdir
import time

def apply_blur(image):
  return image.filter(ImageFilter.GaussianBlur)

def apply_black_and_white(image: Image.Image):
  gray_scale = image.convert('L')
  pixels = list(gray_scale.getdata())
  pixels.sort()
  thresh = pixels[len(pixels) // 2]
  pixels_to_replace = sum(1 for pixel in pixels if pixel < thresh) / len(pixels) * 0.1

  fn = lambda x : 255 if x > thresh else 0
  return gray_scale.point(fn, mode='1'), pixels_to_replace

def apply_noise(image: Image.Image, noise_pxl_count: float):
  noised_image = random_noise(img_as_float(image), mode='s&p', amount=noise_pxl_count)
  return Image.fromarray((noised_image * 255).astype(np.uint8))

def process_image(image_path: str):
  print("Processing image: " + image_path)
  image = Image.open("Images/" + image_path)
  blacked_image, noise_pxl_count = apply_black_and_white(image)
  blacked_image.save("blacked/" + image_path)
  apply_noise(image, noise_pxl_count).save("noised/" + image_path)
  apply_blur(image).save("blurred/" + image_path)

if __name__ == '__main__':
  start = time.perf_counter()
  pool = mp.Pool(mp.cpu_count())

  file_path = "Images/"
  pool.map(process_image, [f for f in listdir(file_path)])

  pool.close()
  pool.join()

# TODO - Add time measurement for each processing function individually
  
  print("Finished in: " + str(time.perf_counter() - start) + " seconds")
  
