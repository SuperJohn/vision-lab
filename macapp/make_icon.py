import cv2, numpy as np
S = 1024; img = np.zeros((S, S, 4), np.uint8)
y = np.linspace(0, 1, S)[:, None]
grad = (np.array([60, 140, 255]) * (1 - y)[..., None] + np.array([195, 124, 142]) * y[..., None]).astype(np.uint8)  # orange->purple (BGR)
mask = np.zeros((S, S), np.uint8); cv2.rectangle(mask, (100, 100), (924, 924), 255, -1)
mask = cv2.GaussianBlur(cv2.dilate(mask, None), (1, 1), 0)
img[..., :3] = np.repeat(grad, S, axis=1); img[..., 3] = 0
r = 180; m = np.zeros((S, S), np.uint8)
cv2.rectangle(m, (100 + r, 100), (924 - r, 924), 255, -1); cv2.rectangle(m, (100, 100 + r), (924, 924 - r), 255, -1)
for cx, cy in [(100 + r, 100 + r), (924 - r, 100 + r), (100 + r, 924 - r), (924 - r, 924 - r)]: cv2.circle(m, (cx, cy), r, 255, -1)
img[..., 3] = m
cv2.ellipse(img, (512, 512), (300, 170), 0, 0, 360, (255, 255, 255, 255), 40, cv2.LINE_AA)
cv2.circle(img, (512, 512), 110, (255, 255, 255, 255), -1, cv2.LINE_AA)
cv2.circle(img, (512, 512), 50, (90, 40, 60, 255), -1, cv2.LINE_AA)
cv2.imwrite("icon.png", img)
