import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

model=YOLO('best.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
            
        



cap=cv2.VideoCapture('cr.mp4')


my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)



count=0
area1=[(544,12),(587,377),(713,372),(643,13)]
area2=[(763,17),(969,343),(1016,298),(924,16)]

while True:    
    ret,frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        w,h=x2-x1,y2-y1
#        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
        cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
        cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
        cvzone.putTextRect(frame,f'person',(x1,y1),1,1)
  
        
   
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
  

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()



