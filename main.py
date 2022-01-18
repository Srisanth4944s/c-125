import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x,y = fetch_openml("mnist_784",version=1,return_x_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
x_train_scaled=x_train/255.0
x_test_scaled=x_test/255.0
clf = LogisticRegression(solver="saga",multi_class="multinomial").fit(x_train_scaled,y_train)
def getPrediction(img):
    im_pil = Image.open(img)
    img_bw = im_pil.convert("L")
    img_bw_rs = img_bw.resize((28,28),Image.ANTIALIAS)
    pixel_flt = 20
    min_pix = np.percentile(img_bw_rs,pixel_flt)
    img_bw_rs_invt_scaled = np.clip(img_bw_rs-min_pix,0,255)
    max_pix = np.max(img_bw_rs)
    img_bw_rs_invt_scaled = np.asarray(img_bw_rs_invt_scaled)/max_pix
    test_sample = np.array(img_bw_rs_invt_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]