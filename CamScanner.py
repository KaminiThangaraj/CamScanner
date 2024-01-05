# import numpy as np 
# import matplotlib.pyplot as plt 
# import cv2
# #%matplotlib inline

# #getting img path and displaying it
# im_path="./bill1.jpg"
# img=cv2.imread(im_path)
# print(img.shape);
# plt.imshow(img)
# plt.show()

# #resizing the image and displaying it
# img=cv2.resize(img,(1500,900))
# print(img.shape);\
# # plt.imshow(img)
# # plt.show()

# #to remove the extra beside the bill we have to follow some steps
# #Remove the noise by blurring  the image
# #Edge dectection
# #Contour Extraction
# #Best Contour Selection

# #Remove the noise by blurring  the image
# org=img.copy()
# gray=cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)
# # plt.imshow(gray,cmap="binary")
# # plt.show()

# blurr = cv2.GaussianBlur(gray,(5,5),0)
# # plt.imshow(blurr,cmap="binary")
# # plt.show()

# regen =cv2.cvtColor(blurr,cv2.COLOR_GRAY2BGR)
# # plt.imshow(regen)
# # plt.show()


# #Edge dectection
# edge = cv2.Canny(blurr,0,50)
# org_edge = edge.copy()
# # plt.imshow(org_edge)
# # plt.title("Edge Detection")
# # plt.show()

# #Contour Extraction
# contour,_=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# print(len(contour))
# contour = sorted(contour,reverse=True,key=cv2.contourArea)

# #Best Contour Selection
# for c in contour:
#     p=cv2.arcLength(c,True)
#     approx=cv2.approxPolyDP(c,0.01*p,True)
#     if len(approx)==4:
#         target = approx
#         break
# print(target.shape)

# def reoder(h):
#     h=h.reshape((4,2))
#     print(h)
#     hnew = np.zeros((4,2),dtype=np.float32)
#     add=h.sum(axis=1)
#     hnew[3]=h[np.argmax(add)]
#     hnew[1]=h[np.argmax(add)]
#     diff = np.diff(h,axis=1)
#     hnew[0]=h[np.argmax(diff)]
#     hnew[2]=h[np.argmax(diff)]
#     return hnew

# reoder = reoder(target)
# print("_________")
# print(reoder)

# in_rep = reoder
# op_map=np.float32([[0,0],[800,0],[800,0],[0,800]])

# M=cv2.getPerspectiveTransform(in_rep,op_map)
# ans = cv2.warpPerspective(org,M,(800,800))
# plt.imshow(ans)
# plt.show()

# # res=cv2.cvtColor(ans,cv2.COLOR_BGR2GRAY)
# # plt.imshow(res,cmap="gray")
# # plt.show()
# res=cv2.cvtColor(ans,cv2.COLOR_GRAY2BGR)
# b_res=cv2.GaussianBlur(res,(3,3),0)
# plt.imshow(b_res,cmap="binary")
# plt.show()

# import numpy as np 
# import matplotlib.pyplot as plt 
# import cv2

# # getting img path and displaying it
# im_path = "./bill1.jpg"
# img = cv2.imread(im_path)
# print("Original Image Shape:", img.shape)
# plt.imshow(img)
# plt.title("Original Image")
# plt.show()

# # resizing the image and displaying it
# img = cv2.resize(img, (1500, 900))
# print("Resized Image Shape:", img.shape)
# plt.imshow(img)
# plt.title("Resized Image")
# plt.show()

# # to remove the extra beside the bill we have to follow some steps
# # Remove the noise by blurring the image
# # Edge detection
# # Contour Extraction
# # Best Contour Selection

# # Remove the noise by blurring the image
# org = img.copy()
# gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
# blurr = cv2.GaussianBlur(gray, (5, 5), 0)

# print("Blurred Image Shape:", blurr.shape)
# plt.imshow(blurr, cmap="binary")
# plt.title("Blurred Image")
# plt.show()

# regen = cv2.cvtColor(blurr, cv2.COLOR_GRAY2BGR)

# # Edge detection
# edge = cv2.Canny(blurr, 0, 50)
# org_edge = edge.copy()

# print("Edge Image Shape:", edge.shape)
# plt.imshow(edge)
# plt.title("Edge Detection")
# plt.show()

# # Contour Extraction
# contour, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# print("Number of Contours:", len(contour))
# contour = sorted(contour, reverse=True, key=cv2.contourArea)

# # Best Contour Selection
# for c in contour:
#     p = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.01 * p, True)
#     if len(approx) == 4:
#         target = approx
#         break

# print("Selected Contour Shape:", target.shape)

# def reorder(h):
#     h = h.reshape((4, 2))
#     print("Original Contour Points:\n", h)
#     hnew = np.zeros((4, 2), dtype=np.float32)
#     add = h.sum(axis=1)
#     hnew[3] = h[np.argmax(add)]
#     hnew[1] = h[np.argmax(add)]
#     diff = np.diff(h, axis=1)
#     hnew[0] = h[np.argmax(diff)]
#     hnew[2] = h[np.argmax(diff)]
#     return hnew

# reordered = reorder(target)
# print("Reordered Contour Points:\n", reordered)

# in_rep = reordered
# op_map = np.float32([[0, 0], [800, 0], [800, 0], [0, 800]])

# M = cv2.getPerspectiveTransform(in_rep, op_map)
# ans = cv2.warpPerspective(org, M, (800, 800))

# print("Warped Image Shape:", ans.shape)
# plt.imshow(ans)
# plt.title("Warped Image")
# plt.show()

# res = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
# print("Grayscale Image Shape:", res.shape)
# plt.imshow(res, cmap="gray")
# plt.title("Grayscale Image")
# plt.show()

# b_res = cv2.GaussianBlur(res, (3, 3), 0)
# print("Blurred Grayscale Image Shape:", b_res.shape)
# plt.imshow(b_res, cmap="binary")
# plt.title("Blurred Grayscale Image")
# plt.show()


# import numpy as np 
# import matplotlib.pyplot as plt 
# import cv2

# # getting img path and displaying it
# im_path = "./bill1.jpg"
# img = cv2.imread(im_path)
# print("Original Image Shape:", img.shape)
# plt.imshow(img)
# plt.title("Original Image")
# plt.show()

# # resizing the image and displaying it
# img = cv2.resize(img, (1500, 900))
# print("Resized Image Shape:", img.shape)
# plt.imshow(img)
# plt.title("Resized Image")
# plt.show()

# # to remove the extra beside the bill we have to follow some steps
# # Remove the noise by blurring the image
# # Edge detection
# # Contour Extraction
# # Best Contour Selection

# # Remove the noise by blurring the image
# org = img.copy()
# gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
# blurr = cv2.GaussianBlur(gray, (5, 5), 0)

# print("Blurred Image Shape:", blurr.shape)
# plt.imshow(blurr, cmap="binary")
# plt.title("Blurred Image")
# plt.show()

# regen = cv2.cvtColor(blurr, cv2.COLOR_GRAY2BGR)

# # Edge detection
# edge = cv2.Canny(blurr, 0, 50)
# org_edge = edge.copy()

# print("Edge Image Shape:", edge.shape)
# plt.imshow(edge)
# plt.title("Edge Detection")
# plt.show()

# # Contour Extraction
# contour, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# print("Number of Contours:", len(contour))
# contour = sorted(contour, reverse=True, key=cv2.contourArea)

# # Best Contour Selection
# for c in contour:
#     p = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.01 * p, True)
#     if len(approx) == 4:
#         target = approx
#         break

# print("Selected Contour Shape:", target.shape)

# def reorder(h):
#     h = h.reshape((4, 2))
#     print("Original Contour Points:\n", h)
#     hnew = np.zeros((4, 2), dtype=np.float32)
#     add = h.sum(axis=1)
#     hnew[3] = h[np.argmax(add)]
#     hnew[1] = h[np.argmax(add)]
#     diff = np.diff(h, axis=1)
#     hnew[0] = h[np.argmax(diff)]
#     hnew[2] = h[np.argmax(diff)]
#     return hnew

# reordered = reorder(target)
# print("Reordered Contour Points:\n", reordered)

# in_rep = reordered
# op_map = np.float32([[0, 0], [800, 0], [800, 0], [0, 800]])

# M = cv2.getPerspectiveTransform(in_rep, op_map)
# ans = cv2.warpPerspective(org, M, (800, 800))

# print("Warped Image Shape:", ans.shape)
# plt.imshow(ans)
# plt.title("Warped Image")
# plt.show()

# res = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
# print("Grayscale Image Shape:", res.shape)
# plt.imshow(res, cmap="gray")
# plt.title("Grayscale Image")
# plt.show()

# b_res = cv2.GaussianBlur(res, (3, 3), 0)
# print("Blurred Grayscale Image Shape:", b_res.shape)
# plt.imshow(b_res, cmap="binary")
# plt.title("Blurred Grayscale Image")
# plt.show()

# # Crop the image
# cropped_img = b_res[100:700, 100:700]
# print("Cropped Image Shape:", cropped_img)
# plt.imshow(cropped_img, cmap="binary")
# plt.title("Cropped Image")
# plt.show()


import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# Getting img path and displaying it
im_path = "./bill1.jpg"  # Replace with the actual path to your image
img = cv2.imread(im_path)
print("Original Image Shape:", img.shape)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()

# Resizing the image
img = cv2.resize(img, (1500, 900))
print("Resized Image Shape:", img.shape)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.show()

# Remove the noise by blurring the image
org = img.copy()
gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
blurr = cv2.GaussianBlur(gray, (5, 5), 0)

print("Blurred Image Shape:", blurr.shape)
plt.imshow(blurr, cmap="binary")
plt.title("Blurred Image")
plt.show()

# Edge detection
edge = cv2.Canny(blurr, 0, 50)
org_edge = edge.copy()

print("Edge Image Shape:", edge.shape)
plt.imshow(edge, cmap="gray")
plt.title("Edge Detection")
plt.show()

# Contour Extraction
contour, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print("Number of Contours:", len(contour))
contour = sorted(contour, reverse=True, key=cv2.contourArea)

# Best Contour Selection
for c in contour:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * p, True)
    if len(approx) == 4:
        target = approx
        break

print("Selected Contour Shape:", target.shape)

def reorder(h):
    h = h.reshape((4, 2))
    print("Original Contour Points:\n", h)
    hnew = np.zeros((4, 2), dtype=np.float32)
    add = h.sum(axis=1)
    hnew[3] = h[np.argmax(add)]
    hnew[1] = h[np.argmax(add)]
    diff = np.diff(h, axis=1)
    hnew[0] = h[np.argmax(diff)]
    hnew[2] = h[np.argmax(diff)]
    return hnew

reordered = reorder(target)
print("Reordered Contour Points:\n", reordered)

in_rep = reordered
op_map = np.float32([[0, 0], [800, 0], [800, 0], [0, 800]])

M = cv2.getPerspectiveTransform(in_rep, op_map)
ans = cv2.warpPerspective(org, M, (800, 800))

print("Warped Image Shape:", ans.shape)
plt.imshow(cv2.cvtColor(ans, cv2.COLOR_BGR2RGB))
plt.title("Warped Image")
plt.show()

res = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
print("Grayscale Image Shape:", res.shape)
plt.imshow(res, cmap="gray")
plt.title("Grayscale Image")
plt.show()

b_res = cv2.GaussianBlur(res, (3, 3), 0)
print("Blurred Grayscale Image Shape:", b_res.shape)
plt.imshow(b_res, cmap="binary")
plt.title("Blurred Grayscale Image")
plt.show()

# Crop the image
cropped_img = b_res[100:700, 100:700]
print("Cropped Image Shape:", cropped_img.shape)
plt.imshow(cropped_img, cmap="binary")
plt.title("Cropped Image")
plt.show()
