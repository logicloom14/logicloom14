import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained model
model_path = r'C:\Users\arron\OneDrive\Documents\UTS\Post Graduate\Semester 2\42174 Artificial Intelligence Studio\Assignment\Week 12\model.h5'
model = load_model(model_path)

# Function to preprocess the image for the model
def preprocess_image(face_img):
    face_img = cv2.resize(face_img, (224, 224))  # Resize to 224x224 based on your model input size
    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    face_img = face_img.astype("float") / 255.0  # Normalize the image
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Function to mark attendance
def mark_attendance(class_id):
    with open('attendance.csv', 'a') as file:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        file.write(f"Class ID: {class_id}, Time: {dt_string}\n")
        print(f"Attendance marked for Class ID: {class_id} at {dt_string}")

# Function to handle login
def login():
    username = entry_username.get()
    password = entry_password.get()
    if username == "admin" and password == "admin":  # Simple check, replace with your own validation
        messagebox.showinfo("Login Info", "Welcome, you have successfully logged in!")
        login_window.destroy()
        open_camera()
    else:
        messagebox.showerror("Login Error", "Invalid username or password")

# Function to open the camera feed
def open_camera():
    camera_window = tk.Tk()
    camera_window.title("Camera Feed")

    label = tk.Label(camera_window)
    label.pack()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        return

    def update_frame():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                processed_face_img = preprocess_image(face_img)
                
                predictions = model.predict(processed_face_img)
                class_id = np.argmax(predictions)
                confidence = np.max(predictions)

                if confidence > 0.5:
                    mark_attendance(class_id)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
        else:
            print("Failed to capture image")

        label.after(10, update_frame)

    update_frame()
    camera_window.mainloop()
    cap.release()

# Create the login window
login_window = tk.Tk()
login_window.title("Login")

# Set the window size
window_width = 300
window_height = 200

# Get the screen dimensions
screen_width = login_window.winfo_screenwidth()
screen_height = login_window.winfo_screenheight()

# Calculate the position for centering the window
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

# Set the geometry of the window
login_window.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# Adding some padding for better layout
padding_options = {'padx': 20, 'pady': 10}

tk.Label(login_window, text="Username").pack(**padding_options)
entry_username = tk.Entry(login_window)
entry_username.pack(**padding_options)

tk.Label(login_window, text="Password").pack(**padding_options)
entry_password = tk.Entry(login_window, show="*")
entry_password.pack(**padding_options)

tk.Button(login_window, text="Login", command=login).pack(**padding_options)

login_window.mainloop()
