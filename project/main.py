from fastapi import FastAPI, Request, Response, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from pydantic import BaseModel
import io,os

from file2 import detect, train_model

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
global_user_id = 1
newuser = False
recog = False
img_id = 0
images_collected = 0
video_capture = None

app = FastAPI()



def collect_user_images():
    global global_user_id, newuser, img_id, images_collected, video_capture
    global_user_id += 1

    if video_capture: 
        video_capture.release()
        cv2.destroyAllWindows()
    # Capturing real-time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
    video_capture = cv2.VideoCapture(0)

    # Initialize img_id with 0
     # Counter for images collected

    while True:  # Capture only 20 images
        ret, img = video_capture.read()
        # Call method we defined above
        if newuser:
            if img_id % 1 == 0:
                print("Collected ", images_collected, " images")
        # Reading image from video stream
            if not ret:  # Check if frame was successfully captured
                print("Error: Failed to capture frame from video stream")
                break
            img = detect(img, faceCascade, img_id, global_user_id)
            # Writing processed image in a new window
            # cv2.imshow("face detection", img)
            print("hello")
            img_id += 1
            images_collected += 1
            if images_collected == 20:
                newuser = False
                train_model()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            newuser = False
            # releasing web-cam
            video_capture.release()
            # Destroying output window
            cv2.destroyAllWindows()
            break

        ret, buffer = cv2.imencode('.png', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

def recognize_faces():
    global recog, video_capture
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # model_file = r"C:\Users\HP\PycharmProjects\pythonProject\trained_model.yml"
    if video_capture: 
        video_capture.release()
        cv2.destroyAllWindows()
    model_file = "trained_model.yml"

    if not os.path.exists(model_file):
        print(f"Error: File '{model_file}' does not exist.")
        return

    recognizer.read(model_file)

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start real-time video capture
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 100 (0 is perfect match)
            if confidence < 100:
                label = "User " + str(id)
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                label = "Unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(frame, str(label), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or recog:
            video_capture.release()
            cv2.destroyAllWindows()
            break
        ret, buffer = cv2.imencode('.png', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    else:
        recog = False



def read(file):
    with open(file, 'r') as f:
        content = f.read()
    return content

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route to stream webcam feed
@app.get('/video_feed')
async def video_feed():
    global newuser
    func = collect_user_images
    return StreamingResponse(func(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/recognise_feed')
async def recognise_feed():
    global recog
    func = recognize_faces
    return StreamingResponse(func(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get('/bg')
async def background():
    with open('./assets/bg.jpg', 'rb') as image_file:
        image_data = image_file.read()
    return Response(content=image_data, media_type='image/jpeg')

class MyData(BaseModel):
    text: str
    
# Post
@app.post("/select",tags=['select'])
async def post(data: MyData):
    global newuser, recog, img_id, images_collected

    text = data.json()
    text = text[len('{"text:" '):- len('"}')]
    print(text)
    if text=="new user":
        recog = False
        newuser = True
        img_id = 0
        images_collected = 0
    elif text=="recog":
        newuser = False
        if recog: recog = False
        else: recog = True
        print(recog)
    return {
            "status":"Success",
            "reply": "ok"
            }
# # Post
# @app.post("/type",tags=['type'])
# async def post(data: MyData):
#     global type_
#     text = data.json()
#     type_ = text[len('{"text:" '):- len('"}')]
#     print(type_)
#     return {
#             "status":"Success",
#             "reply": "ok"
#             }
# @app.post("/mode",tags=['mode'])
# async def post(data: MyData):
#     global mode
#     text = data.json()
#     mode = text[len('{"text:" '):- len('"}')]
#     print(mode)
#     return {
#             "status":"Success",
#             "reply": "ok"
#             }

    
# # Route to serve HTML page
@app.get('/')
async def index(request: Request):
    content = read("index.html")
    return Response(content, media_type='text/html')

# Route to serve HTML page
@app.get('/home')
async def index(request: Request):
    content = read("home.html")
    return Response(content, media_type='text/html')

@app.get('/recognise')
async def index(request: Request):
    content = read("recognise.html")
    return Response(content, media_type='text/html')

# @app.post("/image")
# async def upload_image(image: UploadFile = File(...)):
#     contents = await image.read()
#     result, confidence = cataract.getImg(io.BytesIO(contents))
#     print(confidence)
#     return {"result": result, "confidence": confidence}
