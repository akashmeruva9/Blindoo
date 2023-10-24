import cv2 as cv
import pandas as pd
import bard
import speech_recognition
import re
import pyaudio
import pyttsx3

engine = pyttsx3.init("sapi5")
voices = engine.getProperty('voices')
engine.setProperty('voices', voices[0].id)

known_distance = 20

# Distance constants

object_widths = {
    "person": 18,
    "bicycle": 24,
    "car": 64,
    "motorbike": 36,
    "aeroplane": 108,
    "bus": 96,
    "train": 100,
    "truck": 96,
    "boat": 72,
    "traffic light": 12,
    "fire hydrant": 12,
    "stop sign": 24,
    "parking meter": 12,
    "bench": 24,
    "bird": 4,
    "cat": 10,
    "dog": 12,
    "horse": 36,
    "sheep": 18,
    "cow": 36,
    "elephant": 120,
    "bear": 36,
    "zebra": 36,
    "giraffe": 36,
    "backpack": 12,
    "umbrella": 24,
    "handbag": 12,
    "tie": 6,
    "suitcase": 24,
    "frisbee": 12,
    "skis": 6,
    "snowboard": 12,
    "sports ball": 8,
    "kite": 36,
    "baseball bat": 2,
    "baseball glove": 12,
    "skateboard": 8,
    "surfboard": 36,
    "tennis racket": 12,
    "bottle": 3,
    "wine glass": 4,
    "cup": 3,
    "fork": 1,
    "knife": 1,
    "spoon": 1,
    "bowl": 6,
    "banana": 1,
    "apple": 2,
    "sandwich": 4,
    "orange": 2,
    "broccoli": 3,
    "carrot": 1,
    "hot dog": 1,
    "pizza": 12,
    "donut": 4,
    "cake": 12,
    "chair": 24,
    "sofa": 96,
    "pottedplant": 12,
    "bed": 60,
    "diningtable": 48,
    "toilet": 18,
    "tvmonitor": 24,
    "laptop": 15,
    "mouse": 3,
    "remote": 2,
    "keyboard": 18,
    "cell phone": 3,
    "microwave": 18,
    "oven": 24,
    "toaster": 12,
    "sink": 24,
    "refrigerator": 36,
    "book": 6,
    "clock": 12,
    "vase": 6,
    "scissors": 3,
    "teddy bear": 12,
    "hair drier": 3,
    "toothbrush": 1
}



# Object detector constant
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
        engine.setProperty('rate', 175)
        return True
    except:
        t = "Sorry I couldn't understand and handle this input"
        print(t)
        return False

def takeCommand():
        r = speech_recognition.Recognizer()
        mic = speech_recognition.Microphone()
        with mic as source:
            print("Listening...")
            speak("Listening")

            r.pause_threshold = 0.7
            audio = r.listen(source)

        try:
            print("Recognizing...")
            speak("Recognizing")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
        except Exception:
            speak("Sorry I didn't get that could u repeat")
            print("Sorry I didn't get that could u repeat...")
            return "None"
        return query
# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []

    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 55:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif score > 0.7:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])

        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


def other_distance(real_object_width,width_in_frame):
    distance = (real_object_width * 1300 ) / (width_in_frame)
    return distance


# reading the reference image from dir
ref_person = cv.imread('ReferenceImages/image1.png')
ref_mobile = cv.imread('ReferenceImages/image2.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

# print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length
focal_person = focal_length_finder(known_distance,object_widths["person"], person_width_in_rf)

focal_mobile = focal_length_finder(known_distance,object_widths["cell phone"], mobile_width_in_rf)
cap = cv.VideoCapture(0)


objects = []
obj_distances = []

def scanenv():
    while True:
        ret, frame = cap.read()
        data = object_detector(frame)
        for d in data:
            objects.append(d[0])
            if d[0] == 'person':
                distance = distance_finder(focal_person, object_widths["person"], d[1])
                obj_distances.append(distance)
                x, y = d[2]
            elif d[0] == 'cell phone':
                distance = distance_finder(focal_mobile, object_widths["cell phone"], d[1])
                obj_distances.append(distance)

                x, y = d[2]
            else:
                distance = other_distance(object_widths[d[0]],d[1])
                obj_distances.append(distance)
                x,y = d[2]

            cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

        cv.imshow('frame', frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            cv.destroyAllWindows()
            cap.release()
            break

while True:
   user_cmd = takeCommand()
   if re.search("search",user_cmd):
       scanenv()
       break

obj_distance_map = {"objects":objects,"distance":obj_distances}
df = pd.DataFrame(obj_distance_map)
print('_' * 70)
print("       Average Distance measured(in feet) per object Recognised")
print('_'*70)
grouped_mean_df = df.groupby(["objects"]).mean()/12
print(grouped_mean_df)
print('_' * 70)
print("AI USER ENVIRONMENT DESC")
scene_desc = bard.createSceneFromEnv(grouped_mean_df)
print(scene_desc)

for sentence in scene_desc.split('\n'):
    speak(sentence)
print('_' * 70)
