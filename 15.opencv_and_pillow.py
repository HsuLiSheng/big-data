# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:25:29 2022

@author: user
"""

import cv2

path = 'D:/AI book/introdutction of OpenCV Pillow/images/'

###建立視窗
cv2.namedWindow('Image1')
cv2.namedWindow('Image2')

###預設為讀取彩色圖片，0為讀取灰階圖片
img1 = cv2.imread(path+'img01.jpg')
img2 = cv2.imread(path+'img01.jpg', 0)

cv2.imshow('Image1', img1)
cv2.imshow('Image2', img2)

###等待時間無限長，要按任意鍵繼續執行
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
cv2.namedWindow('Image')

img = cv2.imread(path+'img01.jpg', 0)
cv2.imshow('Image', img)

###設定.jpeg .jpg存檔的格式以及品質(.jpg檔為0~100)
cv2.imwrite(path+'img01copy1.jpg',img)
cv2.imwrite(path+'img01copy2.jpg',img, [int(cv2.IMWRITE_JPEG_QUALITY),50])

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
import numpy as np

cv2.namedWindow('Img')

img = cv2.imread(path+'background.jpg')

###OpenCV顏色分布為(BGR)
###寬度>0:繪圖寬度，<0:實心圖案

###起始點、結束點、顏色、寬度
cv2.line(img, (50,50), (500,200), (255,0,0), 2)

###起始點、結束點、顏色、寬度
cv2.rectangle(img, (100,200), (180,300), (0,255,0), 3)

cv2.rectangle(img, (300,200), (350,260), (0,0,255), -1)

###圓心、半徑、顏色、寬度
cv2.circle(img, (500,300), 40, (255,255,0), -1)

###座標點串列、是否為封閉多邊形、顏色、寬度
pts = np.array([[300,300],[300,340],[350,320]], np.int32)
cv2.polylines(img, [pts], True, (0,255,255), 2)

###文字、位置、字體、字體尺寸、顏色、文字粗細
cv2.putText(img, 'background.jpg', (20,420), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)

cv2.imshow('Img', img)
cv2.imwrite(path+'colored_backgroung.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread(path+'person5.jpg')

faces = facecascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, 
        minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

H = img.shape[0]
W = img.shape[1]

cv2.rectangle(img, (10,H-20), (110,H), (0,0,0), -1)
cv2.putText(img, 'find'+str(len(faces))+'faces!', (10,H-5), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

count=1
for item in faces:
    x,y,w,h = item
    cv2.rectangle(img, (x,y), (x+w,y+h), (128,255,0), 2)
    name = 'face' + str(count) + '.jpg'
    img1 = img[y:y+h, x:x+w]
    img2 = cv2.resize(img1, (400,400))
    cv2.imwrite(path+name, img2)
    count += 1
    
cv2.namedWindow('img')    
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%

img =cv2.imread(path+'person1.jpg')
print('image:',img.shape)

cv2.namedWindow('img1')
cv2.imshow('img1',img)

x,y,w,h = 341,76,125,125
face = img[y:y+h, x:x+w]
face = cv2.resize(face, (400,400))
cv2.imwrite(path+'face.jpg', face)

for row in range(y,y+h):
    for col in range(x,x+w):
        img[row, col][0] = 0
        img[row, col][1] = 50
        
cv2.namedWindow('img2')
cv2.imshow('img2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%

def show(img):
    for y in range(108,114):
        for x in range(6,10):
            print(img[y,x],end=' ')
        print()
    print()
    
img = cv2.imread(path+'face.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('gray image:',img_gray.shape)
show(img_gray)

###cv2.THRESH_BINARY:大於threshold值，設為maxVal，其餘為0
_,thres1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
print('thres1 image:', thres1.shape)
show(thres1)

###cv2.THRESH_BINARY_INV:大於threshold值，設為0，其餘為maxVal
_,thres2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
print('thres2 image:', thres2.shape)
show(thres2)


#%%

###uint8:8位元的int(值在0~255之間)
canvas = np.ones((200,250,3), dtype='uint8')
###設定每一個點的像素質
canvas[:] = (125,40,255)
cv2.imshow('canvas',canvas)


bp = np.zeros((200,250,1), dtype='uint8')
###背景由黑轉白
bp.fill(255)

###把圖片變成白->黑的漸層色
for j in range(200):
    for i in range(250):
        bp[j][i].fill(255-i)
        
cv2.imshow('bp', bp)

cv2.waitKey(0)
cv2.destroyAllWindows()


#%%
import glob

path1 = 'D:/AI book/introdutction of OpenCV Pillow/'
files = glob.glob(path1+'cropMono/*.jpg')
print(files[0])
X = 10
Y = 8
offset = 1

img = cv2.imread(files[0])
H = img.shape[0]
W = img.shape[1]

###合併牌照時要預留邊界的寬度(X、Y)以及字元間的間距(offset)
bg = np.zeros((H+2*Y, (W+offset)*len(files)+2*X, 1), dtype='uint8')
bg.fill(255)
 
for i,file in enumerate(files):
    gray = cv2.imread(file, 0)
    _,thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)    
    
    for row in range(H):
        for col in range(W):
            bg[Y+row][X+col+(W+offset)*i] = thres[row][col]
            
cv2.imwrite(path+'merge.jpg', bg)            


merge = cv2.imread(path+'merge.jpg')   
cv2.imshow('merge', merge)

cv2.waitKey(0)
cv2.destroyAllWindows()


#%%

###0:為筆電的內建攝影機
cap = cv2.VideoCapture(0)

###確認攝影機是否屬於開啟狀態
while(cap.isOpened()):
    cond, img = cap.read()
    if cond == True:
        cv2.imshow('frame', img)
        k = cv2.waitKey(100)
        if k == ord('Z') or k == ord('z'):
            cv2.imwrite(path+'catch.jpg', img)
            break
        
cap.release()            
cv2.destroyAllWindows()         


#%%

from PIL import Image

img = Image.open(path+'img01.jpg')
img.show()

w,h = img.size
name = img.filename

print('image size:('+str(w)+ ',' +str(h) + ')')
print('image name:', name)


#%%

img = Image.open(path+'img01.jpg')
w,h = img.size

img1 = img.resize((w*2,h))
img1.show()

img1.save(path+'resize01.jpg')

#%%

img = Image.open(path+'img01.jpg')
###轉灰階圖片
img_gray = img.convert('L')

img_gray.show()
img_gray.save(path+'gray01.jpg')


#%%

from PIL import ImageDraw,ImageFont

###建立一個淡灰色的畫布
img = Image.new('RGB', (300,400), 'lightgray')
draw = ImageDraw.Draw(img)

###畫金色的圓，包住橢圓外部矩形的左上角和右下角的座標，outline:外框顏色
draw.ellipse((50,50,250,250), width=3, outline='gold')

###位置為各個點組成的串列
draw.polygon([(100,90),(120,130),(80,130)], fill='brown', outline='red')

draw.polygon([(200,90),(220,130),(180,130)], fill='brown', outline='red')

###矩形的左上角和右下角的座標
draw.rectangle((140,140,160,180), fill='blue', outline='black')

draw.ellipse((100,200,200,220), fill='red')

draw.text((130,280), 'test', fill='yellow')

###匯入自選的字型和大小
self_font = ImageFont.truetype('C:\Windows\Fonts\kaiu.ttf',16)
draw.text((110,320),'我日!!!',fille='red',font=self_font)

img.show()
img.save(path+'face1.png')


#%%
###像素的格式: OpenCV: BGR 、 Pillow: RGB
###資料的格式: OpenCV: array() 、 Pillow: Image
#%%
img = cv2.imread(path+'img01.jpg')
cv2.imshow('OpenCV',img)

###OpenCV 轉 Pillow
image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
image.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%
import numpy as np

img = Image.open(path+'img01.jpg')
img.show()

image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.imshow('OpenCV', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%

###處理黑白圖片的像素

img = cv2.imread(path+'img01.jpg',0)
_,thres = cv2.threshold(img, 99, 255, cv2.THRESH_BINARY)
cv2.imwrite(path+'thres1.jpg', thres)


###Pillow要處理每一個像素(getpixel、putpixel)
image = Image.open(path+'img01.jpg')
w,h = image.size

image = image.convert('L')
for i in range(w):
    for j in  range(h):
       if image.getpixel((i,j)) < 99:
             image.putpixel((i,j),(0))
       else:
             image.putpixel((i,j),(255))
             
image.save(path+'thres2.jpg')


