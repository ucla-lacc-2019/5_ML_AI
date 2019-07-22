import numpy as np
import cv2
import time
import pickle
from tkinter import *

cap = cv2.VideoCapture(0)
root = Tk()
label = Label(root)
label.config(font=("Courier", 800))
origin_model = pickle.load(open("model.pkl","rb"))


while True:
    ret,img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray

    crop_img = gray[0:720, 280:1000]
    cv2.imshow('Real time image', crop_img)
    
    img_test = cv2.resize(crop_img, (28,28))
    for i in range(0,28):
    	for j in range(0,28):
    		if img_test[i,j] >= 180:
    			img_test[i,j] = 1
    		else:
    			img_test[i,j] = 0
    # import pdb; pdb.set_trace()

    img_test = np.array(img_test).astype(np.float64)
    # img_test /= np.max(img_test)
    img_test -= np.mean(img_test)
    # cv2.imshow('Proceesed image', img_test)
    # cv2.waitKey()
    img_test = img_test[np.newaxis,:,:,np.newaxis]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pred_testlabel = origin_model.predict_classes(img_test, batch_size=1, verbose=0)
    print(pred_testlabel)

    label['text'] = pred_testlabel[0]
   
    label.pack()
    root.update_idletasks()
    root.update()

    time.sleep(0.15)

cap.release()
cv2.destroyAllWindows() 