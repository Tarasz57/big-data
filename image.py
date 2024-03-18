from PIL import Image, ImageFilter 
from skimage.util import random_noise, img_as_float
import numpy as np

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

if __name__ == '__main__':
  image = Image.open(r"Images/667626_18933d713e.jpg") 
  blacked_image, noise_pxl_count = apply_black_and_white(image)
  blacked_image.save("blacked_image.jpg")
  noised_image = apply_noise(image, noise_pxl_count)
  noised_image.save("noised_image.jpg")
