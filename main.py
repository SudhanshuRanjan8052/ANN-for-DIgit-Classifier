from PIL import Image
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

img = Image.open(r"filepath")

data = list(img.getdata())
for i in range(len(data)):
    data[i]=255-data[i]

input = np.array(data)/256
model = joblib.load("model.sav")

res = model.predict([input])
print("The predicted value is: ",res)
