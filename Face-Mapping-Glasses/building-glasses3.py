import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'D:\\BANGKIT\\CAPSTONE\\wajah/'
img = plt.imread(path + "img8.jpeg")
plt.figure(1)
plt.imshow(img)

img1 = img.copy()

cascade_path = 'D:\\BANGKIT\\CAPSTONE\\frontalEyes35x16.xml'
eye_cascade = cv2.CascadeClassifier(cascade_path)
eye = eye_cascade.detectMultiScale(img)
eye_x, eye_y, eye_w, eye_h = eye[0]

img = cv2.rectangle(img, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 255), 2)
plt.figure(2)
plt.imshow(img)

glasses_filter = plt.imread("D:\\BANGKIT\\CAPSTONE\\kacamata\\10.png")

plt.figure(3)
plt.imshow(glasses_filter)

# Proporsi dan perhitungan relatif
resize_factor_w = (eye_w + 50) / glasses_filter.shape[1]  # Sesuaikan dengan kebutuhan
resize_factor_h = (eye_h + 15) / glasses_filter.shape[0]  # Sesuaikan dengan kebutuhan

# Resize kacamata sesuai dengan faktor perubahan ukuran mata
glasses_filter_updated = cv2.resize(glasses_filter, None, fx=resize_factor_w, fy=resize_factor_h)

plt.figure(4)
plt.imshow(glasses_filter_updated)

# Perubahan pada gambar muka menggunakan kacamata yang diresize
index_0 = glasses_filter_updated.shape[0]
index_1 = glasses_filter_updated.shape[1]
for i in range(index_0):
    for j in range(index_1):
        if (glasses_filter_updated[i, j, 3] > 0):
            img1[int(eye_y + i - 15), int(eye_x + j - 15), :] = glasses_filter_updated[i, j, :-1]

plt.figure(5)
plt.imshow(img1)

# Tampilkan semua figur secara bersamaan
plt.show()
