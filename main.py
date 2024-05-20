import tkinter as tk
import cv2
from PIL import Image, ImageTk
from charm_functions import capture_image, generate_face_embeddings, test_image , InceptionResNetV1
import joblib
import numpy as np
from scipy.spatial.distance import cosine
# from tqdm import tqdm
from keras.models import load_model
import pyfirmata
import time
model = joblib.load('datas/svm_classifier.sav')
label_encoder = joblib.load('datas/label_encoder.sav')

board = pyfirmata.Arduino('COM4')

it = pyfirmata.util.Iterator(board)
it.start()


button_pin = board.get_pin('d:3:i')                      
green_pin = board.get_pin('d:4:o')
red_pin = board.get_pin('d:6:o')
buzzer = board.get_pin('d:8:o')


def button_press():
    button_value = button_pin.read()
    if button_value:
        unlock()
    root.after(100, button_press)  # Check again after 100 milliseconds

def light_green():
    green_pin.write(1)
    buzzer.write(1)
    time.sleep(0.5)
    buzzer.write(0)
    green_pin.write(0)
    


def light_red():
    red_pin.write(1)
    buzzer.write(1)
    time.sleep(0.5)
    buzzer.write(0)
    time.sleep(0.5)
    buzzer.write(1)
    time.sleep(0.5)
    buzzer.write(0)
    red_pin.write(0)


model = joblib.load('datas/svm_classifier.sav')
label_encoder = joblib.load('datas/label_encoder.sav')

known_face_data = np.load('datas/data.npz')
known_face_embeddings = known_face_data['a']

# Load the face recognition model
# Initializing the model.
embeddings_generator = InceptionResNetV1(
    input_shape=(None, None, 3),
    classes=128,
)
# Loading the prebuilt weights.
embeddings_generator.load_weights('facenet_keras_weights.h5')

# Flag to indicate if a face is currently being recognized
face_recognized = False


def unlock():
    global face_recognized
    if face_recognized:
        verify_status.config(text="Verified")
        light_green()
        verify_button.config(bg="green")  # Change button color to green
    else:
        verify_status.config(text="Unverified")
        light_red()
        verify_button.config(bg="red")  # Change button color to red



def recognize_faces(frame, threshold=0.7):
    global face_recognized
    
    # Detect faces in the frame
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face for recognition
        face = cv2.resize(face, (160, 160))
        face = (face.astype('float32') - 127.5) / 128.0  # Normalize to the range [-1, 1]
        face = np.expand_dims(face, axis=0)
        
        # Generate face embeddings using the pre-trained model
        face_embedding = embeddings_generator.predict(face).flatten()
        
        # Calculate cosine similarity between the face and known faces
        similarities = [1 - cosine(face_embedding.flatten(), known_face.flatten()) for known_face in known_face_embeddings]
        
        # Find the maximum similarity
        max_similarity = max(similarities)
        
        # Verify if the face is similar enough to any known face
        if max_similarity >= threshold:
            face_recognized = True
            # Get the label corresponding to the most similar known face
            predicted_label = label_encoder.classes_[np.argmax(similarities)]
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            face_recognized = False
    
    return frame

def capture_and_verify():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = recognize_faces(frame)
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)
    panel.imgtk = frame
    panel.config(image=frame)
    panel.after(10, capture_and_verify)

# Initialize tkinter window
root = tk.Tk()
root.title("Face Recognition")
root.geometry("1000x1000")
root.configure(bg="black")  

# Calculate the center of the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - 1000) // 2
y_coordinate = (screen_height - 1000) // 2

# Center the window on the screen
root.geometry(f"1000x1000+{x_coordinate}+{y_coordinate}")

# Create a frame for the camera feed
camera_frame = tk.Frame(root, width=3000, height=3000, bg="black")
camera_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
cap = cv2.VideoCapture(0)

# Create a label to display the camera feed
panel = tk.Label(camera_frame)
panel.pack(fill=tk.BOTH, expand=True)

# Create a frame for the verification button
button_frame = tk.Frame(root, width=1000, height=80, bg="black")
button_frame.place(relx=0.5, rely=0.85, anchor=tk.CENTER)
verify_button = tk.Button(button_frame, text="Verify", command=unlock, bg="black", fg="white", width=100)
verify_button.pack(fill=tk.BOTH, expand=True)

# Create a frame for the verification status
status_frame = tk.Frame(root, width=1000, height=50, bg="black")
status_frame.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
verify_status = tk.Label(status_frame, text="", font=("Arial", 14), bg="black", fg="white", width=100)
verify_status.pack(fill=tk.BOTH, expand=True)


# Start the camera feed
capture_and_verify()
root.after(100, button_press)

# Run the tkinter event loop
root.mainloop()
