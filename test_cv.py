import cv2
import streamlit as st
import time
import mediapipe as mp




VISUALIZE_FACE_POINTS = False

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

nose_landmarks = [49, 279, 197, 2, 5]



st.title("Webcam Live")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

def gen():

	
	cam = cv2.VideoCapture(0)

	while run:
		ret, frame = cam.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		results = faceMesh.process(rgb)
		
		if not results.multi_face_landmarks:
			print('No face detected ')
			return 0
		
		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				mpDraw.draw_landmarks(frame, face_landmarks, mpFaceMesh.FACEMESH_CONTOURS, drawing_spec)
				
				for lm_id, lm in enumerate(face_landmarks.landmark):
					h, w, c = rgb.shape
					
					x, y = int(lm.x * w), int(lm.y * h)
					
					if lm_id in nose_landmarks:
						cv2.putText(frame, str(lm_id), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
						
		FRAME_WINDOW.image(frame)
		
	else:
		st.write('Stopped')
gen()		
