import os
import shutil
import dlib
import cv2
import numpy as np
import customtkinter as ctk
import threading
import time
from tkinter import filedialog, messagebox, StringVar

# Load the models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1(
    "./dlib_face_recognition_resnet_model_v1.dat"
)

# Variables to manage the state of the filtering process
stop_thread = False
start_time = None
thread = None

known_image_path = None
images_folder = None
filtered_images_folder = None


def get_face_encodings(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    face_encodings = []
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        face_encoding = face_rec_model.compute_face_descriptor(image, landmarks)
        face_encodings.append(np.array(face_encoding))
    return face_encodings


def is_match(known_encodings, unknown_encodings, threshold=0.5):
    for unknown_encoding in unknown_encodings:
        distances = [
            np.linalg.norm(unknown_encoding - known_encoding)
            for known_encoding in known_encodings
        ]
        if min(distances) < threshold:
            return True
    return False


def get_all_image_files(base_path):
    jpg_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                jpg_files.append(os.path.join(root, file))
    return jpg_files


def upload_known_image():
    global known_image_path
    known_image_path = filedialog.askopenfilename(
        title="Select Known Image", filetypes=[("Image files", "*.jpg;*.jpeg")]
    )
    if known_image_path:
        lbl_known_image.configure(text=os.path.basename(known_image_path))


def select_images_folder():
    global images_folder
    images_folder = filedialog.askdirectory(title="Select Folder to Filter")
    if images_folder:
        lbl_images_folder.configure(text=images_folder)


def select_filtered_images_folder():
    global filtered_images_folder
    filtered_images_folder = filedialog.askdirectory(
        title="Select Folder to Save Filtered Images"
    )
    if filtered_images_folder:
        lbl_filtered_images_folder.configure(text=filtered_images_folder)


def start_filtering():
    global stop_thread, start_time
    stop_thread = False
    start_time = time.time()

    if not known_image_path or not images_folder or not filtered_images_folder:
        messagebox.showwarning("Warning", "Please select all paths.")
        return

    # Reset and show progress bar and labels, and stop button
    progress_bar.set(0)
    progress_bar.pack(pady=20)
    percentLabel.pack()
    taskLabel.pack()
    btn_stop.pack(pady=5)

    # Show status message
    statusLabel.pack(pady=5)
    root.update_idletasks()

    # Simulate initialization delay
    time.sleep(2)  # Adjust this delay as needed

    # Remove status message
    statusLabel.pack_forget()

    thread = threading.Thread(target=filter_images)
    thread.start()


def filter_images():
    global stop_thread
    known_encodings = get_face_encodings(known_image_path)
    jpg_files = get_all_image_files(images_folder)

    total_files = len(jpg_files)
    progress_bar.set(0)
    percent.set(f"{1 / total_files * 100:.2f}%")
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_text.set(f"Time Elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    for index, unknown_image_path in enumerate(jpg_files):
        if stop_thread:
            break
        unknown_encodings = get_face_encodings(unknown_image_path)
        if is_match(known_encodings, unknown_encodings):
            shutil.copy(unknown_image_path, filtered_images_folder)
        progress_bar.set((index + 1) / total_files)
        percent.set(f"{(index + 1) / total_files * 100:.2f}%")
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_text.set(
            f"Time Elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        )
        root.update_idletasks()

    if not stop_thread:
        messagebox.showinfo("Info", "Filtering Complete!")
    else:
        messagebox.showinfo("Info", "Filtering Stopped!")

    # Hide progress bar and labels after completion or stopping
    progress_bar.set(0)
    progress_bar.pack_forget()
    percent.set("0%")
    percentLabel.pack_forget()
    time_text.set("Time Elapsed 00:00:00")
    taskLabel.pack_forget()
    btn_stop.pack_forget()


def stop_filtering():
    global stop_thread, thread
    stop_thread = True
    if thread is not None:
        thread.kill()
        thread.join()  # Wait for the thread to finish

    # Hide progress bar and labels immediately when stopping
    progress_bar.set(0)
    progress_bar.pack_forget()
    percent.set("0%")
    percentLabel.pack_forget()
    time_text.set("00:00:00")
    taskLabel.pack_forget()
    btn_stop.pack_forget()


def on_closing():
    global stop_thread, thread
    stop_thread = True
    if thread is not None:
        thread.kill()
        thread.join()  # Wait for the thread to finish
    stop_thread = True
    root.destroy()


# Set up the customtkinter window
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Face Recognition Filter")

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the desired window size (e.g., 50% of screen size)
window_width = int(screen_width * 0.6)
window_height = int(screen_height * 0.6)

# Set window size and position
root.geometry(
    f"{window_width}x{window_height}+100+100"
)  # +100+100 sets the window position

heading = ctk.CTkLabel(
    root, text="Face Recognition Filter", font=ctk.CTkFont(size=20, weight="bold")
)
heading.pack(pady=10)

# Frame for file selection
frame_file_selection = ctk.CTkFrame(root)
frame_file_selection.pack(pady=10)

# Upload the known image
btn_known_image = ctk.CTkButton(
    frame_file_selection, text="Upload Known Image", command=upload_known_image
)
btn_known_image.grid(row=0, column=0, padx=10, pady=5)
lbl_known_image = ctk.CTkLabel(frame_file_selection, text="No file selected", width=40)
lbl_known_image.grid(row=0, column=1, padx=10, pady=5)

# Select the folder to filter
btn_images_folder = ctk.CTkButton(
    frame_file_selection, text="Select Folder to Filter", command=select_images_folder
)
btn_images_folder.grid(row=1, column=0, padx=10, pady=5)
lbl_images_folder = ctk.CTkLabel(
    frame_file_selection, text="No folder selected", width=40
)
lbl_images_folder.grid(row=1, column=1, padx=10, pady=5)

# Select the folder to save filtered images
btn_filtered_images_folder = ctk.CTkButton(
    frame_file_selection,
    text="Select Folder to Save Filtered Images",
    command=select_filtered_images_folder,
)
btn_filtered_images_folder.grid(row=2, column=0, padx=10, pady=5)
lbl_filtered_images_folder = ctk.CTkLabel(
    frame_file_selection, text="No folder selected", width=40
)
lbl_filtered_images_folder.grid(row=2, column=1, padx=10, pady=5)

# Start the filtering process
btn_start = ctk.CTkButton(
    root, text="Start Filtering", command=start_filtering, width=20
)
btn_start.pack(pady=10)

# Stop button (initially hidden)
btn_stop = ctk.CTkButton(root, text="Stop Filtering", command=stop_filtering, width=20)
btn_stop.pack_forget()

# Progress bar and labels (initially hidden)
progress_bar = ctk.CTkProgressBar(root)
percent = StringVar()
time_text = StringVar()
percentLabel = ctk.CTkLabel(root, textvariable=percent)
taskLabel = ctk.CTkLabel(root, textvariable=time_text)

# Status label (initially hidden)
status_label_text = StringVar()
status_label_text.set("Initializing image filtration...")
statusLabel = ctk.CTkLabel(root, textvariable=status_label_text, fg_color="blue")

# Bind the on_closing function to the window close event
root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
