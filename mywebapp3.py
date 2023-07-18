import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp

#-----------------------------------------------
import os
from twilio.rest import Client
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)
token = client.tokens.create()
#-----------------------------------------------

hands = mp.solutions.hands.Hands(max_num_hands=2)

font = cv2.FONT_HERSHEY_SIMPLEX

st.title("ตรวจจับการชูนิ้ว")

class VideoProcessor:  
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #----------------------------------------------
        img = cv2.flip(img,1) 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        h, w, c = img.shape # 640 480 3
        
        if results.multi_hand_landmarks:
            numfingerLeft = 0
            numfingerRight = 0
            
            for handLms in results.multi_hand_landmarks:
                
                handIndex = results.multi_hand_landmarks.index(handLms)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                point8 = handLms.landmark[8]
                cx8, cy8 = int(point8.x*w), int(point8.y*h)
                cv2.circle(img,(cx8,cy8),8,(0,255,255),-1)

                point12 = handLms.landmark[12]
                cx12, cy12 = int(point12.x*w), int(point12.y*h)
                cv2.circle(img,(cx12,cy12),8,(0,255,255),-1)

                point16 = handLms.landmark[16]
                cx16, cy16 = int(point16.x*w), int(point16.y*h)
                cv2.circle(img,(cx16,cy16),8,(0,255,255),-1)
                
                point20 = handLms.landmark[20]
                cx20, cy20 = int(point20.x*w), int(point20.y*h)
                cv2.circle(img,(cx20,cy20),8,(0,255,255),-1)

                point5 = handLms.landmark[5]
                point5.y = point5.y-0.05
                cx5, cy5 = int(point5.x*w), int(point5.y*h)
                cv2.circle(img,(cx5,cy5),8,(0,255,0),-1)
                    
                if handLabel == 'Left':
                    if  (cy8<cy5):
                        numfingerLeft += 1
                    if  (cy12<cy5):
                        numfingerLeft += 1
                    if  (cy16<cy5):
                        numfingerLeft += 1
                    if  (cy20<cy5):
                        numfingerLeft += 1
                        
                    cv2.putText(img,str(numfingerLeft),(100,100),font,2,(255,0,0),10)
                    cv2.putText(img,handLabel,(100,200),font,2,(255,0,0),10)
                    
                else:
                    
                    if  (cy8<cy5):
                        numfingerRight += 1
                    if  (cy12<cy5):
                        numfingerRight += 1
                    if  (cy16<cy5):
                        numfingerRight += 1
                    if  (cy20<cy5):
                        numfingerRight += 1  
                        
                    cv2.putText(img,str(numfingerRight),(int(w/2),100),font,2,(255,0,0),10)
                    cv2.putText(img,handLabel,(int(w/2),200),font,2,(255,0,0),10)

                '''
                if  (cy8>cy5) and (cy12>cy5) and (cy16>cy5) and (cy20>cy5):
                    cv2.putText(img,"0",(100,200),font,5,(255,0,0),10)
                elif  (cy8<cy5) and (cy12>cy5) and (cy16>cy5) and (cy20>cy5):
                    cv2.putText(img,"1",(100,200),font,5,(255,0,0),10)               
                elif  (cy8<cy5) and (cy12<cy5) and (cy16>cy5) and (cy20>cy5):
                    cv2.putText(img,"2",(100,200),font,5,(255,0,0),10)
                elif (cy8<cy5) and (cy12<cy5) and (cy16>cy5) and (cy20>cy5):
                    cv2.putText(img,"3",(100,200),font,5,(255,0,0),10)
                elif  (cy8<cy5) and (cy12<cy5) and (cy16<cy5) and (cy20>cy5):
                    cv2.putText(img,"4",(100,200),font,5,(255,0,0),10)
                '''
        #----------------------------------------------
            cv2.putText(img,str(numfingerLeft+numfingerRight),(int((w/2)-50),250),font,2,(255,0,0),10)
             
        return av.VideoFrame.from_ndarray(img,format="bgr24")

webrtc_streamer(key="test",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True,"audio": False})
