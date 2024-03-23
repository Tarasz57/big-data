from multiprocessing.managers import ListProxy
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
  blacked_image = gray_scale.point(fn, mode='1')
  return blacked_image, pixels_to_replace


def apply_noise(image: Image.Image, noise_pxl_count: float):
  noise = random_noise(img_as_float(image), mode='s&p', amount=noise_pxl_count)
  return Image.fromarray((noise * 255).astype(np.uint8))


def find_average(times: ListProxy):
  return sum(times) / len(times)


def process_image(image_path: str, times_to_bw: list, times_to_noise: list, times_to_blur: list):
  print("Processing image: " + image_path)
  image = Image.open("Images/" + image_path)

  time_to_bw = time.perf_counter()
  blacked_image, noise_pxl_count = apply_black_and_white(image)
  time_to_bw = time.perf_counter() - time_to_bw
  blacked_image.save("blacked/" + image_path)

  time_to_noise = time.perf_counter()
  noised_image = apply_noise(image, noise_pxl_count)
  time_to_noise = time.perf_counter() - time_to_noise
  noised_image.save("noised/" + image_path)

  time_to_blur = time.perf_counter()
  blurred_image = apply_blur(image)
  time_to_blur = time.perf_counter() - time_to_blur
  blurred_image.save("blurred/" + image_path)

  times_to_bw.append(time_to_bw)
  times_to_noise.append(time_to_noise)
  times_to_blur.append(time_to_blur)


if __name__ == '__main__':
  times_to_bw = mp.Manager().list()
  times_to_noise = mp.Manager().list()
  times_to_blur = mp.Manager().list()
  pool = mp.Pool(mp.cpu_count())

  file_path = "Images/"
  image_count = len(listdir(file_path))
  start = time.perf_counter()
  
  pool.starmap(process_image, [(f, times_to_bw, times_to_noise, times_to_blur ) for f in listdir(file_path)])

  end = time.perf_counter() - start
  pool.close()
  pool.join()
  
  print("Finished processing of " + str(image_count)+ " images" + " in: " + str(end) + " seconds")
  print("Average time to black and white: " + str(find_average(times_to_bw)) + " seconds")
  print("Average time to noise: " + str(find_average(times_to_noise)) + " seconds")
  print("Average time to blur: " + str(find_average(times_to_blur)) + " seconds")
  
