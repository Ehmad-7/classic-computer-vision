import cv2
import matplotlib.pyplot as plt

image_path='images/sample.jpg'
image=cv2.imread(image_path)

print('Image Shape: ',image.shape)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

resized=cv2.resize(gray,(1288,1288))

plt.imshow(resized,cmap='gray')
plt.axis('off')
plt.show()