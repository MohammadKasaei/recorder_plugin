
# 🧩 Recorder Plugin (`recorder_plugin`) — v0.0.0

The `RecorderPlugin` (`recorder_plugin`) is an RQt plugin for real-time video recording and playback in a ROS-based context. It captures image frames, logs control signals (e.g., cable positions, Z-translation), and replays recorded data for debugging, training, or analysis.

---

## 📦 Features

| Feature | Description |
|--------|-------------|
| **Real-Time Recording** | Captures frames from a connected video device or stream and saves them with cable positions and translation data. |
| **Replay Control** | Replays recorded image sequences with GUI-based pause/resume and speed control. |
| **Dynamic Frame Cropping & Rotation** | Crops and rotates frames on-the-fly for better alignment and focus. |
| **Frame Grabber** | Allows live image acquisition from `/dev/video*` devices with optional trimming of dark borders. |
| **Config Persistence** | Saves and loads frame grabber settings (crop, rotation, FPS, etc.) between sessions. |
| **UI Integration** | Complete RQT GUI using `recorder.ui` with PyQt5 controls. |


# Install QT5
Use the following command to install QT5 and related tools:
<!-- bash command -->

sudo apt install python3-pyqt5 qttools5-dev-tools qtcreator pyqt5-dev-tools

---

## 🧠 How It Works

### 1. **Recording Mode**
- When the user clicks "Record", the plugin:
  - Subscribes to `/read_camera_image/image_bronch`, `/cables_pos`, and `/z_translation`.
  - Saves incoming image frames with time-stamped sensor values to disk.

### 2. **Replay Mode**
- When "Replay" is clicked:
  - Images from the selected folder are played back at the specified speed.
  - Frames are published back to `/read_camera_image/image_bronch` for downstream visualization.

### 3. **Frame Grabber**
- Captures real-time images using OpenCV from selected video devices.
- Applies optional border trimming, rotation, and cropping.
- Publishes images over ROS topics.

### 4. **Configurable UI Elements**
- Speed, crop region, rotation angle, and threshold ratio can be configured via sliders and saved as persistent configuration.

---

## ⚙️ Main Functions

| Method | Purpose |
|--------|---------|
| `handle_btn_record_click()` | Starts/stops video and sensor recording |
| `handle_btn_replay_click()` | Starts/stops replay of previously recorded image sequences |
| `handle_btn_frm_grabber_click()` | Starts/stops live frame grabbing from camera or video stream |
| `on_grab_frame_timer_timeout()` | Grabs, processes, and publishes live images |
| `load_frm_grabber_cfg()` / `handle_btn_save_cfg_click()` | Loads and saves persistent configuration |
| `smart_trim_black_borders()` | Trims unnecessary black borders from camera input |

---

## 🚀 Installation

Clone this repository into your ROS workspace's `src/` directory:

```bash
cd ~/catkin_ws/src
git clone https://github.com/MohammadKasaei/recorder_plugin.git
```
  
Install dependencies:

```bash
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

Make sure you have the following ROS dependencies installed:

```bash
sudo apt install ros-${ROS_DISTRO}-rqt-gui-py
sudo apt install ros-${ROS_DISTRO}-cv-bridge
sudo apt install ros-${ROS_DISTRO}-image-transport
```

---
## Launch Instructions

To run the plugin within an RQt environment:

```bash
rqt
```

Then, navigate to `Plugins > bronchoscopy_plugin > MyPlugin` to open the GUI.

## 🖼️ UI

The plugin uses a `.ui` file (`recorder.ui`) which should be placed under `resource/` inside the ROS package. It must define buttons, sliders, labels, and image viewers for full functionality.
## 💾 Recorded Data

During recording, the following files are saved in the selected folder:
- `image_XXXX.jpg` — Captured image frames.
- `cables_pos.txt` — Logged cable position data (timestamped).
- `z_translation.txt` — Logged Z-axis translation values.

---