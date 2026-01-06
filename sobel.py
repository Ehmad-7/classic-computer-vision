import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/sample.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Sobel X")
plt.imshow(sobel_x, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Sobel Y")
plt.imshow(sobel_y, cmap="gray")
plt.axis("off")

plt.show()
