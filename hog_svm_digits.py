import numpy as np
import cv2
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

digits=load_digits()
print(digits.data.shape)

x=digits.data
y=digits.target

hog_features=[]

for img in x:
    img=img.astype("uint8") 
    img=cv2.resize(img,(32,32))
    features=hog(
        img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm="L2-Hys"
    )
    hog_features.append(features)

hog_features=np.array(hog_features)

x_train,x_test,y_train,y_test=train_test_split(
    hog_features,y,test_size=0.2,random_state=42
)

classifier=svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

predictions=classifier.predict(x_test)

accuracy=accuracy_score(y_test,predictions)
print("HOG + SVM Accuracy: ",accuracy)
