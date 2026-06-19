import cv2
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
from os.path import join
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN


class Application:
    # initialization
    def __init__(self):
        
        # face detector
        self.face_detector = MTCNN(min_face_size=30)

        # define the face input size
        self.input_size = (160, 160, 3)
        self.image_size = (480, 640, 3)

        # local variables
        self.video_stream = cv2.VideoCapture(0)
        self.is_running = False
        self.time_delay = 30

        # setup GUI
        self.gui_window = tk.Tk()
        self.gui_window.wm_title('Face Detection Demo')
        self.gui_window.config(background='#FFFFFF')

        # construct the GUI
        self.create_widgets()

    # create widgets (button, panel, etc)
    def create_widgets(self):
        # create menu
        self.menu_bar = tk.Menu(self.gui_window)
        self.gui_window.config(menu=self.menu_bar)

        # create file menu
        self.menu_file = tk.Menu(self.menu_bar)
        self.menu_file.add_command(label="Exit", command=self.exit_app)
        self.menu_bar.add_cascade(label="File", menu=self.menu_file)

        # create setting menu
        self.menu_edit = tk.Menu(self.menu_bar)
        self.menu_edit.add_command(label="Database")
        self.menu_bar.add_cascade(label="Setting", menu=self.menu_edit)

        # create help menu
        self.menu_help = tk.Menu(self.menu_bar)
        self.menu_bar.add_cascade(label="Help", menu=self.menu_help)

        # video stream title
        self.video_title = tk.Label(self.gui_window, 
            text='Video Stream Preview')
        self.video_title.grid(row=0, column=0, padx=5, pady=2)

        # graphics window
        self.frame_holder = tk.Frame(self.gui_window, width=650, height=500)
        self.frame_holder.grid(row=1, column=0, padx=10, pady=2)

        # frame window
        self.frame_panel = tk.Label(self.frame_holder)
        self.frame_panel.grid(row=1, column=0)
        self.get_initial_frame()

        # button
        self.process_button = tk.Button(self.gui_window, width=10, height= 2)
        self.process_button.grid(row=2, column=0, padx=10, pady=2)
        self.process_button['text'] = 'START'
        self.process_button['command'] = self.start_app
    
    # exit app
    def exit_app(self):
        exit()

    # capture frame
    def get_frame(self):
        # read frame from webcam
        is_read, frame = self.video_stream.read()
        if is_read:
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
            frame = cv2.flip(frame, 1)
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
            result_img = rgb_img.copy() # frame for drawing 
            
            # detect face
            face_data = self.face_detector.detect_faces(rgb_img)

            # if face is exist
            if len(face_data) > 0:                  
                # draw rectangle of face region    
                for n in range(len(face_data)):
                    x, y, w, h = face_data[n]['box']
                    start_point = (x, y)
                    end_point = (x + w, y + h)
                    result_img = cv2.rectangle(result_img, start_point, end_point, (255, 0, 0), 2)

            # display the result
            rgba_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2RGBA)    
            rgb_array = Image.fromarray(rgba_img)
            rgb_imgtk = ImageTk.PhotoImage(image=rgb_array)
            self.frame_panel.imgtk = rgb_imgtk
            self.frame_panel.configure(image=rgb_imgtk)

            # check if still running
            if self.is_running:
                self.frame_panel.after(10, self.get_frame)

    # initialize empty frame
    def get_initial_frame(self):
        # display the initial frame
        white_img = np.ones(self.image_size, dtype=np.uint8)
        white_img = white_img*128
        rgba_img = cv2.cvtColor(white_img, cv2.COLOR_RGB2RGBA)    
        rgb_array = Image.fromarray(rgba_img)
        rgb_imgtk = ImageTk.PhotoImage(image=rgb_array)
        self.frame_panel.imgtk = rgb_imgtk
        self.frame_panel.configure(image=rgb_imgtk)

        # check if not running
        if not self.is_running:
            self.frame_panel.after(10, self.get_initial_frame)

    # start/stop button
    def start_app(self):
        # if user press start
        self.is_running = not self.is_running
        if self.is_running:
            self.process_button['text'] = 'STOP'
            if not self.video_stream.isOpened():
                self.video_stream.open(0)
            self.get_frame()    
        # if user press stop   
        else:
            self.process_button['text'] = 'START'
            if self.video_stream.isOpened:
                self.video_stream.release()
            self.get_initial_frame()

app = Application()
app.gui_window.mainloop()