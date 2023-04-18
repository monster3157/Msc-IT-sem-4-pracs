#https://learndataanalysis.org/source-code-getting-started-with-microsoft-azure-face-api-in-python/
#pip3 install azure.cognitiveservices.vision.face
import os
import io
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests
from PIL import Image, ImageDraw, ImageFont

"""
Example 2. Detect faces landmarks from an image (from a local file)
"""

API_KEY = "d6257ff7f426"
ENDPOINT = "https[end point]ices.azure.com/"
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

img_file = open('image1.jpg', 'rb')

response_detected_faces = face_client.face.detect_with_stream(
    image=img_file,
    detection_model='detection_03',
    recognition_model='recognition_04',
    return_face_landmarks=True
)

if not response_detected_faces:
    raise Exception('No face detected')

print('Number of people detected: {0}'.format(len(response_detected_faces)))

print(vars(response_detected_faces[0]))
print(vars(response_detected_faces[0].face_landmarks).keys())
print(response_detected_faces[0].face_landmarks.mouth_left)

img =Image.open(img_file)
draw = ImageDraw.Draw(img)

for face in response_detected_faces:
    rect = face.face_rectangle
    left = rect.left
    top = rect.top
    right = rect.width + left
    bottom = rect.height + top
    draw.rectangle(((left, top), (right, bottom)), outline='green', width=5)

img.show()
