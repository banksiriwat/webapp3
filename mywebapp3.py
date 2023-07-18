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

face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

font = cv2.FONT_HERSHEY_SIMPLEX

st.title("หันซ้าย หันขวา")

class VideoProcessor:  
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #--------------------------------------------------
        img = cv2.flip(img,1)
        
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(imgRGB)

        h, w, c = img.shape

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                
                point137 = faceLms.landmark[137]   
                cx137, cy137 = int(point137.x*w), int(point137.y*h)
                cv2.circle(img,(cx137,cy137),10,(0,0,255),-1)

                point4 = faceLms.landmark[4]   
                cx4, cy4 = int(point4.x*w), int(point4.y*h)
                cv2.circle(img,(cx4,cy4),10,(0,255,0),-1)

                point366 = faceLms.landmark[366]   
                cx366, cy366 = int(point366.x*w), int(point366.y*h)
                cv2.circle(img,(cx366,cy366),10,(255,0,0),-1)

                cv2.line(img,(cx4,cy4),(cx137,cy137),(0,0,255),5)
                cv2.line(img,(cx4,cy4),(cx366,cy366),(255,0,0),5)

                distL = int(cx4-cx137)
                distR = int(cx366-cx4)            
                 
                if (distL > distR) and (distL - distR) > 40 :
                    cv2.putText(img,"Right",(50,100),font,2,(0,180,100),8)
                elif (distR > distL) and (distR - distL) > 40 :
                    cv2.putText(img,"Left",(50,100),font,2,(0,100,255),8)
                else:
                    cv2.putText(img,"Straight",(50,100),font,2,(255,0,0),8)
        #--------------------------------------------------
        return av.VideoFrame.from_ndarray(img,format="bgr24")

webrtc_streamer(key="test",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True,"audio": False},
                rtc_configuration={"iceServers":token.ice_servers}) ###


