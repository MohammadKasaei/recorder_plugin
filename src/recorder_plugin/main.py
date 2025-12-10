# src/my_rqt_plugin/my_module.py

import os
from random import random
import rospy
import rospkg
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QApplication
from rqt_gui_py.plugin import Plugin

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
from PIL import Image as PILImage
import PIL
import numpy as np  
import torchvision.transforms as transforms

from std_msgs.msg import Float64, Float64MultiArray

from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QGraphicsScene
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtCore import Qt, QTimer, Signal, Slot
from rqt_gui_py.plugin import Plugin
from python_qt_binding.QtWidgets import QGraphicsPixmapItem

import torch
from pathlib import Path
        
import subprocess
import fnmatch
import glob


class MyPlugin(Plugin):

    # Define a signal for thread-safe GUI updates
    image_signal = Signal(QImage)
    
    def __init__(self, context):
        super(MyPlugin, self).__init__(context)
        self.setObjectName('MyPlugin')
        self.bridge = CvBridge()
        self._inital_rotation = 0
        self._inital_crop = [158,0,500,480]
        self.reset_landmarks = False
        self._save_landmark = False
        self._update_landmarks = False
        self._replay_folder_path = ""
        self.frame_count = 0 
        self._frame_per_second = 1
        self._pause = False
        self._rec_pause = False
        self._can_save = False
        
        self._thr_ratio = 30
        self.data_str = ""
        self._tool_trans_data_str = ""
        self._internal_cnt = 0
        self._replay_repeat_count = 1

        # Create QWidget
        self._widget = QWidget()

        # Get path to UI file
        ui_file = os.path.join(
            rospkg.RosPack().get_path('recorder_plugin'),
            'resource',
            'recorder.ui')

        # Load the UI file
        loadUi(ui_file, self._widget)
        self._widget.setObjectName('RecorderUi')

        # Add widget to the user interface
        if context.serial_number() > 1:
            self._widget.setWindowTitle(
                self._widget.windowTitle() +
                (' (%d)' % context.serial_number()))
        context.add_widget(self._widget)
        
        home = Path.home()
        self.frm_cfg_save_path = f"{home}/frame_grabber.cfg"
        self.load_frm_grabber_cfg()

        self._widget.btn_replay.clicked.connect(self.handle_btn_replay_click)
        self._widget.btn_pause.clicked.connect(self.handle_btn_pause_click)
        self._widget.btn_record.clicked.connect(self.handle_btn_record_click)
        self._widget.btn_rec_pause.clicked.connect(self.handle_btn_rec_pause_click)
        self._widget.btn_frm_grabber.clicked.connect(self.handle_btn_frm_grabber_click)
        self._widget.btn_save_cfg.clicked.connect(self.handle_btn_save_cfg_click)
        self._widget.btn_gen_video.clicked.connect(self.handle_btn_gen_video_click)
        self._widget.btn_open_path.clicked.connect(self.handle_btn_open_path_click)
          
        
        
        self._widget.tbr_replay_spd.valueChanged.connect(self.on_tbr_replay_spd_changed)   
        self._widget.tbr_replay_repeat.valueChanged.connect(self.on_tbr_replay_repeat_changed)
        self._widget.tbr_replay_progress.valueChanged.connect(self.on_tbr_replay_progress_changed)
        self._widget.tbr_rec_spd.valueChanged.connect(self.on_tbr_rec_spd_changed)   
        self._widget.tbr_frm_grab_spd.valueChanged.connect(self.on_tbr_frm_grab_spd_changed)
        self._widget.tbr_frm_grab_rot.valueChanged.connect(self.on_tbr_frm_grab_rot_changed)
        self._widget.tbr_thr_ratio.valueChanged.connect(self.on_tbr_thr_ratio_changed)
        
        
        # add 0 to 5 to the combo box
        device_names = self.get_video_device_names()
        
        if len(device_names) > 0:
            self._widget.cmb_frm_grab_src.addItems(device_names)
        else:
            self._widget.cmb_frm_grab_src.addItems([str(i) for i in range(6)])
        
        
        value = self._widget.tbr_replay_spd.value()
        self._widget.lbl_replay_spd.setText(f"Speed:\t{value} hz")
        value = self._widget.tbr_rec_spd.value()
        self._widget.lbl_rec_spd.setText(f"Skip Frame:\t{value-1}")
        value = self._widget.tbr_frm_grab_spd.value()
        self._widget.lbl_frm_grab_spd.setText(f"Speed:\t{value} FPS")
        
        self._widget.tbr_frm_grab_rot.setValue(self._inital_rotation)
        self._widget.lbl_frm_grab_rot.setText(f"Rot.:\t{self._inital_rotation} deg")
        self._frame_grabber_rotation = value
        
        
        
        self._widget.btn_replay.setText("Replay")
        self._widget.btn_replay.setStyleSheet("color: green;")
        
        self._widget.btn_pause.setStyleSheet("color: red;")
        self._widget.btn_pause.setEnabled(False)
        
        self._widget.btn_record.setText("Record")
        self._widget.btn_record.setStyleSheet("color: green;")
        
        
        self._widget.btn_rec_pause.setStyleSheet("color: red;")
        self._widget.btn_rec_pause.setEnabled(False)
        
        self._widget.btn_frm_grabber.setStyleSheet("color: green;")
        
        self._widget.txt_frm_grab_crop.setPlainText(f"{self._inital_crop[0]},{self._inital_crop[1]},{self._inital_crop[2]},{self._inital_crop[3]}")
        
            
        # Initialize the timer
        self.replay_timer = QTimer(self)
        self.frm_grabber_timer = QTimer(self)
    

    def loginfo(self, msg: str, ID = 0):
        if ID == 0:
            rospy.loginfo(f"[Recorder - {ID}] {msg}")
        elif ID == 1:
            rospy.logerr(f"[Recorder - {ID}] {msg}")

    def handle_btn_open_path_click(self):
        # open a dialog to select a folder
        from PyQt5.QtWidgets import QFileDialog
        folder_path = QFileDialog.getExistingDirectory(self._widget, "Select Folder")
        if folder_path:
            self._widget.txt_path_replay.setPlainText(folder_path)
        
    
    def get_video_device_names(self):
        devices = []
        try:
            output = subprocess.check_output("ls /dev/video* ", shell=True).decode("utf-8")
            self.loginfo(f"output : {output}")
            lines = output.splitlines()
            for line in lines:
                if line and "/dev/video" in line:
                    devices.append(line)
                
        except Exception as e:
            self.loginfo(f"Error: {e}", 1)

        return devices

    def handle_btn_record_click(self):
        try:
            if self._widget.btn_record.text() == "Record":
                self.save_path = self._widget.txt_path_record.toPlainText()
                chk_new_folder = self._widget.chk_new_folder.isChecked()
                if self.save_path == "":
                    self._widget.btn_record.setText("Record")
                    self._widget.btn_record.setStyleSheet("color: green;")
                    self.loginfo(f"saving path is not correct.", 1)
                    return
                
                if chk_new_folder:    
                    # Create a new folder to save the images based on time and date
                    self.save_path = os.path.join(self.save_path, time.strftime("%Y%m%d-%H%M%S"))
                    # Create the directory
                    os.makedirs(self.save_path)
                    self._widget.txt_path_record.setPlainText(self.save_path)
                    self.loginfo(f"New folder created: {self.save_path}")

                # Open text files for other data
                self.cables_pos_file = open(os.path.join(self.save_path, "cables_pos.txt"), "a")
                self.tool_translation_file = open(os.path.join(self.save_path, "tool_translation.txt"), "a")
                self.image_sub = rospy.Subscriber("/read_camera_image/image_bronch", Image, self.image_callback)
                
                self.cables_pos_sub = rospy.Subscriber("/cables_pos", Float64MultiArray, self.cables_pos_callback)
                self.tool_translation_sub = rospy.Subscriber("/tool_translation", Float64MultiArray, self.tool_translation_callback)
                self.frame_count = 0
                self._frame_saved_count = 0
        
                self._widget.btn_record.setText("Stop")
                self._widget.btn_record.setStyleSheet("color: red;")
                self._widget.lbl_record.setText("Recording...")
                style_sheet = f"color: red;"
                self._widget.lbl_record.setStyleSheet(style_sheet)
                
                self._widget.btn_rec_pause.setEnabled(True)
                self._rec_pause = False
                self._can_save = True                
                
            else:
                self.image_sub.unregister()
                self.cables_pos_sub.unregister()
                self.tool_translation_sub.unregister()
                self.cables_pos_file.close()
                self.tool_translation_file.close()
                self._can_save = False
                
                self._widget.btn_record.setText("Record")
                self._widget.btn_record.setStyleSheet("color: green;")
                self._rec_pause = False
                self._widget.btn_rec_pause.setEnabled(False)
                self._widget.lbl_record.setText("---")
                style_sheet = f"color: black;"
                self._widget.lbl_record.setStyleSheet(style_sheet)
                
        except Exception as e:
            self.loginfo(f"Failed to record the images: {e}", 1)
            return
        
    
    def cables_pos_callback(self, msg):
        if not self._can_save:
            return

        self.data_str = ','.join(map(str, msg.data))
        
        # timestamp = time.time()
        # self.cables_pos_file.write(f"{timestamp},{data_str}\n")
        # self.cables_pos_file.flush()  # Ensure data is written immediately

    def tool_translation_callback(self, msg):
        if not self._can_save:
            return
        self._tool_trans_data_str = ','.join(map(str, msg.data))


    def load_frm_grabber_cfg(self):
        try:
            if os.path.exists(self.frm_cfg_save_path):
                with open(self.frm_cfg_save_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        parts = line.split(",")
                        key = parts[0].strip()

                        if len(parts) == 2:
                            if key == "rotation":
                                value = int(parts[1].strip())
                                self._widget.tbr_frm_grab_rot.setValue(value)
                                self._inital_rotation = value
                                self.loginfo(f"rotation: {self._inital_rotation}")
                            elif key == "speed":
                                value = int(parts[1].strip())
                                self._widget.tbr_frm_grab_spd.setValue(value)
                                self.loginfo(f"speed: {value}")
                            elif key == "show_img":
                                value = parts[1].strip()
                                if value == "True":
                                    self._widget.chk_show_img.setChecked(True)
                                else:
                                    self._widget.chk_show_img.setChecked(False)
                                self.loginfo(f"show_img: {value}")
                            elif key == "thr_ratio":
                                value = int(parts[1].strip())
                                self._widget.tbr_thr_ratio.setValue(value)
                                self._thr_ratio = value
                                self._widget.lbl_thr_ratio.setText(f"Ratio:\t{self._thr_ratio}")
                                self.loginfo(f"thr_ratio: {self._thr_ratio}")
                            elif key == "replay_path":
                                value = parts[1].strip()
                                self._widget.txt_path_replay.setPlainText(value)
                                self.loginfo(f"replay_path: {value}")
                            elif key == "publish_cable":
                                value = parts[1].strip()
                                if value == "True":
                                    self._widget.chk_publish_joints.setChecked(True)
                                else:
                                    self._widget.chk_publish_joints.setChecked(False)
                                self.loginfo(f"publish_cable: {value}")
                            elif key == "replay_speed":
                                value = int(parts[1].strip())
                                self._widget.tbr_replay_spd.setValue(value)
                                self.loginfo(f"replay_speed: {value} hz")
                            elif key == "replay_repeat":
                                value = int(parts[1].strip())
                                self._widget.tbr_replay_repeat.setValue(value)
                                self._replay_repeat_count = value
                                self.loginfo(f"replay_repeat: {value}")
                                self._widget.lbl_replay_repeat.setText(f"Repeat:\t{self._replay_repeat_count} times")
                        elif key == "crop":
                            value = f"{parts[1].strip()},{parts[2].strip()},{parts[3].strip()},{parts[4].strip()}"
                            self._widget.txt_frm_grab_crop.setPlainText(value)
                            self._inital_crop = [int(x) for x in value.split(",")]
                            self.loginfo(f"crop: {self._inital_crop}")
                        
                        

                self.loginfo("Frame grabber configuration loaded.")

            else:
                self.loginfo(f"Frame grabber configuration file does not exist.", 1)
        except Exception as e:
            self.loginfo(f"Failed to load the frame grabber configuration: {e}", 1)
            return
        

    def handle_btn_save_cfg_click(self):
        try:
            data = []
            data.append(["rotation",self._widget.tbr_frm_grab_rot.value()])
            data.append(["crop",self._widget.txt_frm_grab_crop.toPlainText()])
            data.append(["speed",self._widget.tbr_frm_grab_spd.value()])
            data.append(["show_img",self._widget.chk_show_img.isChecked()])
            data.append(["thr_ratio",self._widget.tbr_thr_ratio.value()])
            data.append(["replay_path",self._widget.txt_path_replay.toPlainText()])
            data.append(["publish_cable", self._widget.chk_publish_joints.isChecked()])
            data.append(["replay_speed", self._widget.tbr_replay_spd.value()])
            data.append(["replay_repeat", self._widget.tbr_replay_repeat.value()])

            with open(self.frm_cfg_save_path, 'w') as file:
                for item in data:
                    # Convert each item to string and write to the file
                    file.write(f"{item[0]}, {item[1]}\n")

            self.loginfo("Configuration saved to: {}".format(self.frm_cfg_save_path))

        except Exception as e:
            self.loginfo(f"Failed to save the configuration: {e}", 1)
            return
    
    
    def handle_btn_gen_video_click(self):
        try:
            self._widget.btn_gen_video.setStyleSheet("color: red;")
            self.generate_video()
            self._widget.btn_gen_video.setStyleSheet("color: green;")
        except Exception as e:
            self.loginfo(f"Failed to generate the video: {e}", 1)
            return
    
    def generate_video(self):
        try:
            # generate mp4 video from the images
            
            self._replay_folder_path = self._widget.txt_path_replay.toPlainText()
            
            images = sorted(glob.glob(os.path.join(self._replay_folder_path, "*.jpg")))
            if not images:
                self.loginfo("No PNG images found in the folder.", 1)
                return
            
            # Read the first image to get frame size
            frame = cv2.imread(images[0])
            height, width, layers = frame.shape
            len_images = len(images)
            self.loginfo(f"Generating video from the images...")
            fps = self._widget.tbr_frm_grab_spd.value()
            self._widget.lbl_replay.setText(f"Gen ... 0/{len_images}")
            self._widget.btn_gen_video.setStyleSheet("color: red;")
            self._widget.lbl_replay.setStyleSheet("color: red;")
            
            video_path = os.path.join(self._replay_folder_path, "video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for i, file in enumerate(images):
                if i % 1000 == 0:
                    self.loginfo(f"Generating video from the images... {i+1}/{len_images}")
                    self._widget.lbl_replay.setText(f"Gen: {i+1}/{len_images}")
                    QApplication.processEvents()

                # self.loginfo(f"Generating video from the images... {i+1}/{len(images)}")
                img = PILImage.open(file)
                cv_image = np.array(img)
                out.write(cv_image)
            out.release()
            self._widget.lbl_replay.setText("---")
            self._widget.lbl_replay.setStyleSheet("color: black;")
            self.loginfo(f"Video generated successfully.")
        except Exception as e:
            self.loginfo(f"Failed to generate the video: {e}", 1)
            return
    
    def handle_btn_frm_grabber_click(self):
        if self._widget.btn_frm_grabber.text() == "Start":
            try:
                
                self.replay_timer.stop()
                self._widget.btn_pause.setText("Pause")
                self._widget.btn_pause.setEnabled(False)
                
                self._widget.btn_replay.setText("Replay")
                self._widget.btn_replay.setStyleSheet("color: green;")
                self._widget.lbl_replay.setText("---")
                style_sheet = f"color: black;"
                self._widget.lbl_replay.setStyleSheet(style_sheet)
                
                self._inital_rotation = self._widget.tbr_frm_grab_rot.value()
                crop_str = self._widget.txt_frm_grab_crop.toPlainText()
                self._inital_crop = [int(x) for x in crop_str.split(",")]
                if len(self._inital_crop) != 4:
                    self.loginfo(f"Invalid crop values.", 1)
                    self._cropping = False
                else:
                    self._cropping = True
                    
                    
                # Open the capture card device (0 or 1 depending on your system)
                # get the current index
                index = self._widget.cmb_frm_grab_src.currentIndex()
                self._cap = cv2.VideoCapture(index)  # or `1` for a second device
                # self._cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
                
                # fps = self._widget.tbr_frm_grab_spd.value()
                # # Define the stream URL (change this to your stream address)
                # stream_url = 'udp://169.254.99.215:1234'  

                # # Open the video stream using OpenCV
                # self._cap = cv2.VideoCapture(stream_url,cv2.CAP_FFMPEG)
                # # self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                # # self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                # self._cap.set(cv2.CAP_PROP_FPS, fps)


                # Check if the capture is successful
                if not self._cap.isOpened():
                    self.loginfo("Error: Could not open video device", 1)
                    return

                # Capture a single frame
                ret, frame = self._cap.read()
                
                if ret:
                    self.loginfo(f"Image captured {frame.shape}")
                    img = PILImage.fromarray(frame)
                    if self._cropping:
                        img = transforms.functional.rotate(img,self._inital_rotation).crop(self._inital_crop)
                    else:
                        img = transforms.functional.rotate(img,self._inital_rotation)
                        
                    cv_image = np.array(img)
                    self._widget.lbl_frm_grab_img_size.setText(f"Size:\t{frame.shape[1]}x{frame.shape[0]} -> {cv_image.shape[1]}x{cv_image.shape[0]}")
                else:
                    self.loginfo("Error: Failed to capture image", 1)
                    return


                
                self._widget.btn_frm_grabber.setText("Stop")
                if self._widget.chk_show_img.isChecked():
                    cv2.namedWindow("frame grabber", cv2.WINDOW_NORMAL)

                fps = self._widget.tbr_frm_grab_spd.value()
                self.frm_grabber_timer.setInterval(1000/fps)  # Set interval to 5000 ms (5 seconds)
                self.frm_grabber_timer.timeout.connect(self.on_grab_frame_timer_timeout)  # Connect timer to a method
                self.replay_image_pub = rospy.Publisher('/read_camera_image/image_bronch', Image, queue_size=1)
                
                self.cnt = 0
                self._widget.btn_frm_grabber.setStyleSheet("color: red;")
                self.frm_grabber_timer.start()
            except Exception as e:
                self.loginfo(f"Failed to start the frame grabber: {e}", 1)
                self._widget.btn_frm_grabber.setText("Start")
                self._widget.btn_frm_grabber.setStyleSheet("color: green;")
                return
        else:
            self._widget.btn_frm_grabber.setText("Start")
            self.frm_grabber_timer.stop()
            self._widget.btn_frm_grabber.setStyleSheet("color: green;")
            # Release the capture device
            self._cap.release()
                

    def smart_trim_black_borders(self, img, black_thresh=10, min_content_thresh_ratio=30):
        """
        Aggressively trims dark borders from an image. Logos or text in dark areas will be removed.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        min_row_thresh = w * min_content_thresh_ratio
        min_col_thresh = h * min_content_thresh_ratio

        # Threshold the image to identify non-black (bright) areas
        _, mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY)

        # Get row and column indices with sufficient white pixels
        rows = np.where(np.sum(mask, axis=1) > min_row_thresh)[0]
        cols = np.where(np.sum(mask, axis=0) > min_col_thresh)[0]

        if len(rows) == 0 or len(cols) == 0:
            return img  # probably all black or too dark

        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]

        return img[top:bottom+1, left:right+1]

        
    def on_grab_frame_timer_timeout(self):
        
        try:
            ret, frame = self._cap.read()
            if not ret:
                self.loginfo("Failed to grab the frame...", 1)
                return
            # self.loginfo (f"{self.cnt}frame shape: {frame.shape}")
            # self.cnt +=1
            if self._thr_ratio > 0:
                frame = self.smart_trim_black_borders(frame,min_content_thresh_ratio=self._thr_ratio)
            
            img = PILImage.fromarray(frame) 
            if self._cropping:
                img = transforms.functional.rotate(img,self._inital_rotation).crop(self._inital_crop)
            else:
                img = transforms.functional.rotate(img,self._inital_rotation)
                
            cv_image = np.array(img)
            if self._widget.chk_show_img.isChecked():
               cv2.imshow("frame grabber", cv_image)
            
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.replay_image_pub.publish(image_msg)
            
        except Exception as e:
            self.loginfo(f"Failed to grab the frame: {e}", 1)
            self.frm_grabber_timer.stop()
            self._widget.btn_frm_grabber.setText("Start")
            self._widget.btn_frm_grabber.setStyleSheet("color: green;")
            
            return
        
    def handle_btn_rec_pause_click(self):
        if self._widget.btn_rec_pause.text() == "Pause":
            self._rec_pause = True
            self._widget.btn_rec_pause.setText("Resume")
            self._widget.btn_rec_pause.setStyleSheet("color: green;")
        else:
            self._rec_pause = False
            self._widget.btn_rec_pause.setText("Pause")
            self._widget.btn_rec_pause.setStyleSheet("color: red;")
            
    def handle_btn_pause_click(self):
        if self._widget.btn_pause.text() == "Pause":
            self._pause = True
            self._widget.btn_pause.setText("Resume")
            self._widget.btn_pause.setStyleSheet("color: green;")
        else:
            self._pause = False
            self._widget.btn_pause.setText("Pause")
            self._widget.btn_pause.setStyleSheet("color: red;")
            
            
    def on_replay_timer_timeout(self):
        """
        This method will be called every time the timer times out.
        You can update the label or perform other actions here.
        """
        
        if self._replay_folder_path == "":
            self.loginfo(f"replay path is not defined.", 1)
            # Stop the timer
            self.replay_timer.stop()
            self._widget.btn_replay.setText("Replay")
            self._widget.btn_replay.setStyleSheet("color: green;")
            return
        try:
            if self._pause:
                return
            
            
            image_path = os.path.join(self._replay_folder_path, f"image_{self._replay_frame_count:04d}.jpg")
            img = PILImage.open(image_path)
            cv_image = np.array(img)
            
            # check the shape first and if needed resize to 600 x 600 in case the image is not
            if self._widget.chk_random_mask.isChecked():
                # add random black mask to the image
                h, w, _ = cv_image.shape
                mask_size = int(min(h, w) * 0.2)
                top_left_x = np.random.randint(0, w - mask_size)
                top_left_y = np.random.randint(0, h - mask_size)
                cv_image[top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = 0
                
            if self._widget.chk_random_motion_jitter.isChecked(): 
                
                # add random motion jitter to the image
                jitter_amount = 20  # pixels
                jitter_x = np.random.randint(-jitter_amount, jitter_amount)
                jitter_y = np.random.randint(-jitter_amount, jitter_amount)
                M = np.float32([[1, 0, jitter_x], [0, 1, jitter_y]])
                cv_image = cv2.warpAffine(cv_image, M, (cv_image.shape[1], cv_image.shape[0]))
                
                # random rotation
                angle = np.random.uniform(-30, 30)  # degrees
                center = (cv_image.shape[1] // 2, cv_image.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cv_image = cv2.warpAffine(cv_image, M, (cv_image.shape[1], cv_image.shape[0]))
                
                # random bending
                offset = np.random.uniform(-60, 60)
                h, w = cv_image.shape[:2]
                src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                dst_pts = np.float32([
                    [0 + offset, 0],
                    [w - offset, 0],
                    [0 - offset, h],
                    [w + offset, h]
                ])
                M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
                cv_image = cv2.warpPerspective(cv_image, M_persp, (w, h),
                                        borderValue=(255, 255, 255))

            if self._widget.chk_random_color_jitter.isChecked(): 
                # add random color jitter to the image
                hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
                hue_shift = np.random.randint(-10, 10)
                sat_scale = 1 + (np.random.rand() - 0.5) * 0.4  # Scale between 0.8 and 1.2
                val_scale = 1 + (np.random.rand() - 0.5) * 0.4  # Scale between 0.8 and 1.2

                hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
                hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * sat_scale, 0, 255)
                hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * val_scale, 0, 255)

                cv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
                
            if self._widget.chk_random_distortion.isChecked(): 
                h, w = cv_image.shape[:2]

                amplitude = np.random.uniform(5, 10)
                frequency = np.random.uniform(20, 20)

                x, y = np.meshgrid(np.arange(w), np.arange(h))

                x_distorted = x + amplitude * np.sin(2 * np.pi * y / frequency)
                y_distorted = y

                cv_image = cv2.remap(cv_image,
                                x_distorted.astype(np.float32),
                                y_distorted.astype(np.float32),
                                cv2.INTER_LINEAR)

            if self._widget.chk_random_crop.isChecked():
                h, w = cv_image.shape[:2]
                crop_ratio = np.random.uniform(0.7, 1.0)  # Keep 70% to 100% of the image
                new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
                start_x = np.random.randint(0, w - new_w)
                start_y = np.random.randint(0, h - new_h)
                cv_image = cv_image[start_y:start_y + new_h, start_x:start_x + new_w]
                
            if self._widget.chk_random_blur.isChecked():
                # add random blur to the image
                ksize = np.random.choice([3, 5])  # kernel size
                iter = np.random.randint(1, 3)  # number of iterations
                # for _ in range(iter):
                #     cv_image = cv2.GaussianBlur(cv_image, (ksize, ksize), 0)
                cv_image = cv2.medianBlur(cv_image, ksize)
                
            if cv_image.shape[:2] != (600, 600):
                cv_image = cv2.resize(cv_image, (600, 600))
            # cv_image = self.smart_trim_black_borders(cv_image)
            
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.replay_image_pub.publish(image_msg)
            
            if self._widget.chk_publish_joints.isChecked():
                # publish cables and tool states
                cables_msg = Float64MultiArray()
                tool_msg = Float64MultiArray()
                if not hasattr(self, '_cables_data') or not hasattr(self, '_tool_translation_data'):
                    self.loginfo(f"Joints data not loaded.", 1)
                    self.loginfo(f"Loading joints data from the replay folder...")
                    self.load_cables_pos_data_ready_for_publish()

                else:
                    if self._replay_frame_count >= self._cables_data.shape[0]:
                        self.loginfo(f"end of tool data.")
                    else:
                        cables_msg.data = self._cables_data[self._replay_frame_count][1:].tolist()
                        tool_msg.data = self._tool_translation_data[self._replay_frame_count][1:].tolist()
                        self.cables_pos_pub.publish(cables_msg)
                        self.tool_translation_pub.publish(tool_msg)

            if self._internal_cnt % self._replay_repeat_count == 0:
                self._replay_frame_count += 1
                
            self._internal_cnt += 1
            
            self._widget.tbr_replay_progress.setValue(self._replay_frame_count)
            
            
        except Exception as e:
            self.loginfo(f"Failed to check the replay path: {e}", 1)
            # Stop the timer
            self.replay_timer.stop()
            self._widget.lbl_replay.setText("---")
            style_sheet = f"color: black;"
            self._widget.lbl_replay.setStyleSheet(style_sheet)
            self._widget.btn_replay.setText("Replay")
            self._widget.btn_replay.setStyleSheet("color: green;")
            return
    
            
    def on_tbr_thr_ratio_changed(self, value):
        """
        Handle the slider value change event.
        """
        self._widget.lbl_thr_ratio.setText(f"Ratio:\t{value}")
        self._thr_ratio = value
        
    def on_tbr_replay_repeat_changed(self, value):
        """
        Handle the slider value change event.
        """
        
        if value>1:
            self._widget.lbl_replay_repeat.setText(f"Repeat:\t{value} times")
        else:
            self._widget.lbl_replay_repeat.setText(f"Repeat:\t{value} time")
        
        self._replay_repeat_count = value

    def on_tbr_replay_spd_changed(self, value):
        """
        Handle the slider value change event.
        """
        self._widget.lbl_replay_spd.setText(f"Speed:\t{value} hz")
        self.replay_timer.setInterval(1000/value)  
        
    def on_tbr_frm_grab_rot_changed(self, value):
        """
        Handle the slider value change event.
        """
        self._widget.lbl_frm_grab_rot.setText(f"Rot.\t{value} deg")
        self._frame_grabber_rotation = value
        
        
    def on_tbr_frm_grab_spd_changed(self, value):
        """
        Handle the slider value change event.
        """
        self._widget.lbl_frm_grab_spd.setText(f"Speed:\t{value} FPS")
        self.frm_grabber_timer.setInterval(1000/value)
        
    def on_tbr_rec_spd_changed(self, value):
        """
        Handle the slider value change event.
        """
        self._frame_per_second = value
        self._widget.lbl_rec_spd.setText(f"Skip Frame:\t{value-1}")
        
    def on_tbr_replay_progress_changed(self, value):
        """
        Handle the slider value change event.
        """
        self._replay_frame_count = value
        self._widget.tbr_replay_progress.setValue(self._replay_frame_count)
        
    def handle_btn_replay_click(self):
        try:
            if self._widget.btn_replay.text() == "Replay":
                self._replay_folder_path = self._widget.txt_path_replay.toPlainText()
                self._widget.btn_frm_grabber.setText("Start")
                self.frm_grabber_timer.stop()
                self._widget.btn_frm_grabber.setStyleSheet("color: green;")
                # Release the capture device
                # self._cap.release()
                
                if self._replay_folder_path == "":
                    self._widget.btn_replay.setText("Replay")
                    self._widget.btn_replay.setStyleSheet("color: green;")
                    self.loginfo(f"Please load landmarks first.", 1)
                    return
                
                self._widget.btn_replay.setText("Stop")
                self._widget.btn_replay.setStyleSheet("color: red;")
                self._widget.btn_pause.setEnabled(True)
                
            
            
                replay_speed = self._widget.tbr_replay_spd.value()

                self.replay_timer.setInterval(1000/replay_speed)  # Set interval to 5000 ms (5 seconds)
                self.replay_timer.timeout.connect(self.on_replay_timer_timeout)  # Connect timer to a method
                self.replay_image_pub = rospy.Publisher('/read_camera_image/image_bronch', Image, queue_size=10)
                
                files = os.listdir(self._replay_folder_path)
                files = [f for f in files if f.endswith('.jpg')]
                self._widget.tbr_replay_progress.setMaximum(len(files))
                self._widget.tbr_replay_progress.setValue(0)
                self._replay_frame_count = 0

                if self._widget.chk_publish_joints.isChecked():
                    self.load_cables_pos_data_ready_for_publish()
                    # self.cables_pos_pub = rospy.Publisher("/cables_pos", Float64MultiArray, queue_size=10)
                    # self.tool_translation_pub = rospy.Publisher("/tool_translation", Float64MultiArray, queue_size=10)
                    # self._cables_data = np.loadtxt(os.path.join(self._replay_folder_path, "cables_pos.txt"), delimiter=',')
                    # self._tool_translation_data = np.loadtxt(os.path.join(self._replay_folder_path, "tool_translation.txt"),  delimiter=',')
                    # self.loginfo (f"cables_data shape: {self._cables_data.shape}")
                    # self.loginfo (f"tool_translation_data shape: {self._tool_translation_data.shape}")

                # Start the timer
                self.replay_timer.start()
                self._widget.lbl_replay.setText("Replaying")
                style_sheet = f"color: red;"
                self._widget.lbl_replay.setStyleSheet(style_sheet)

                
            else:
                # Stop the timer
                self.replay_timer.stop()
                self._widget.btn_pause.setText("Pause")
                self._widget.btn_pause.setEnabled(False)
                
                self._widget.btn_replay.setText("Replay")
                self._widget.btn_replay.setStyleSheet("color: green;")
                self._widget.lbl_replay.setText("---")
                style_sheet = f"color: black;"
                self._widget.lbl_replay.setStyleSheet(style_sheet)
                
                
        except Exception as e:
            self.loginfo(f"Failed to replay the images: {e}", 1)
            self.replay_timer.stop()
            self._widget.btn_replay.setText("Replay")
            self._widget.btn_replay.setStyleSheet("color: green;")
            
            self._widget.lbl_replay.setText("---")
            style_sheet = f"color: black;"
            self._widget.lbl_replay.setStyleSheet(style_sheet)
            return
        
    def load_cables_pos_data_ready_for_publish(self):
        try:
            self.cables_pos_pub = rospy.Publisher("/cables_pos", Float64MultiArray, queue_size=10)
            self.tool_translation_pub = rospy.Publisher("/tool_translation", Float64MultiArray, queue_size=10)
            self._cables_data = np.loadtxt(os.path.join(self._replay_folder_path, "cables_pos.txt"), delimiter=',')
            self._tool_translation_data = np.loadtxt(os.path.join(self._replay_folder_path, "tool_translation.txt"),  delimiter=',')
            self.loginfo (f"cables_data shape: {self._cables_data.shape}")
            self.loginfo (f"tool_translation_data shape: {self._tool_translation_data.shape}")
        except Exception as e:
            self.loginfo (f"Failed to load the cables position data: {e}", 1)
            return

    def image_callback(self, msg):
        
        self.frame_count += 1
        if self.frame_count % (self._frame_per_second) != 0:
            return
        
        if self._rec_pause:
            return 
        
        # add a try except block to handle the error
        try:
            if self.data_str == "" or self._tool_trans_data_str == "":
                self.loginfo(f"cables position data is not ready yet.")
                # return
                self.data_str = "0.0,0.0,0.0,0.0"
                self._tool_trans_data_str = "0.0, 0.0"
                

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            fname = f"{self.save_path}/image_{self._frame_saved_count:04d}.jpg"
            self._frame_saved_count += 1
            cv2.imwrite(fname, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for OpenCV

            timestamp = time.time()
            self.cables_pos_file.write(f"{timestamp},{self.data_str}\n")
            self.cables_pos_file.flush()  #
        
            timestamp = time.time()
            self.tool_translation_file.write(f"{timestamp},{self._tool_trans_data_str}\n")
            self.tool_translation_file.flush()  # Ensure data is written immediately

        except Exception as e:
            self.loginfo (f"Failed to save the image: {e}", 1)
            return
        

        