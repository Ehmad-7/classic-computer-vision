import cv2
import matplotlib.pyplot as plt

image_path='images/sample.jpg'

image=cv2.imread(image_path)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

blurred=cv2.GaussianBlur(gray,(7,7),0)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Original Grayscale Image')
plt.imshow(gray,cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Blurred Image')
plt.imshow(blurred,cmap='gray')
plt.axis('off')
plt.show()