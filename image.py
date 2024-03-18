

if __name__ == '__main__':
  from PIL import Image, ImageFilter 
  from skimage.util import random_noise, img_as_float
  import numpy as np

  # Opening the image  
  # (R prefixed to string in order to deal with '\' in paths) 
  image = Image.open(r"Images/667626_18933d713e.jpg") 
    
  blur_image = image.filter(ImageFilter.GaussianBlur) 
    
  blur_image.save('blurred.jpg')

  thresh = 150
  fn = lambda x : 255 if x > thresh else 0
  r = image.convert('L').point(fn, mode='1')
  r.save('blacked.png')

  noised_image = random_noise(img_as_float(image), mode='gaussian',mean=0.15)
  Image.fromarray((noised_image * 255).astype(np.uint8)).save('noised.jpg')
