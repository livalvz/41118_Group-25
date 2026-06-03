# Drive Control Installation and Running Instructions
 
A hand gesture controlled driving system using MediaPipe landmark detection and a Random Forest classifier to drive a simulated car in PyBullet.
 
## Clone Repository
1. Generate an SSH key (if you don't have one)
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
2. Add the key to your GitHub account
```bash
# Copy the public key to clipboard
cat ~/.ssh/id_ed25519.pub
Then go to GitHub → Settings → SSH keys → New SSH key, paste it in and save.
```
3. Clone the repo
```bash
git clone git@github.com:livalvz/41118_Group-25.git
cd 41118_Group-25 (you will need to add the full path e.g. cd git/41118_Group-25)
```
 
## Installations
Install these before running
 
**ROS2 Humble (Ubuntu 22.04)**
```bash
sudo apt update
sudo apt install -y   ros-humble-desktop   ros-humble-cv-bridge   ros-humble-vision-opencv   python3-colcon-common-extensions   python3-pip   python3-rosdep
```
 
**Python dependencies**
```bash
pip install   mediapipe   opencv-python   scikit-learn   joblib   "numpy<2"   pandas   flask   flask-socketio   pybullet   pybullet-utils   gymnasium   matplotlib
```
 
## First Time Setup
 
```bash
# Navigate to the template directory
cd ~/git/41118_Group-25 (this will need to be your path to the folder)
 
# Source ROS2
source /opt/ros/humble/setup.bash
 
# Build the workspace
colcon build
 
# Source the workspace
source install/setup.bash
```
 
> You need to run `source /opt/ros/humble/setup.bash` and `source install/setup.bash` in every new terminal before running the system.
 
## Collect
```bash
# Step 1 — collect gesture training data
cd ~/git/41118_Group-25/template (this will need to be your path to the folder)
python3 hand_gesture/collect_landmark.py
 
```
A webcam window will open. Hold up each gesture and press the matching key to save samples.
 
Try to collect samples with:
 
slightly different hand positions
slightly different distances from the camera
the same lighting conditions you will use during the demo
enough samples for every gesture
 
Aim for at least 30 to 50 samples per gesture. The three_fingers gesture is especially important because it is used for manual direction control.
 
The samples will be saved into:
```bash
gesture_landmarks.csv
```
After collecting samples, train the classifier using the next section.
 
## Train the Gesture Classifier
 
```bash
# Step 2 — train the classifier
cd ~/git/41118_Group-25/template (this will need to be your path to the folder)
python3 hand_gesture/train_classifier.py
```
 
## Running the System
 
**Terminal 1 — launch everything**
```bash
cd ~/git/41118_Group-25 (this will need to be your path to the folder)
source /opt/ros/humble/setup.bash
source install/setup.bash
cd ~/git/41118_Group-25/template (this will need to be your path to the folder)
ros2 launch hand_driving driving.launch.py
```
 
Wait about 5-10 seconds for PyBullet to initialise.
 
**Terminal 2 — open the GUI**
```bash
#the terminal will print out a link you can ctrl+press the link to open
"""
the link will be printed out as such
* Running on http://127.0.0.1:5000
* Running on http://192.168.0.218:5000
"""
```
 
## Static Gesture Controls
 
| Gesture | Action |
|---------|--------|
| 1 finger | Drive to the blue block |
| 2 fingers | Drive to red block |
| Open palm | Stop immediately - will remain stopped until a new command |
| Closed fist | Teleport car back to home position |
| Move hand with three fingers up to the left zone| Turn left (while held) |
| Move hand with three fingers up to the right zone | Turn right (while held) |
| Move hand with three fingers up to the top middle zone | Go straight |
 
## Dynamic Gesture Controls
| Three-finger hand position | Command |
|---------|--------|
|Three fingers moving to the left zone| Turn left |
|Three fingers moving to the right zone| Turn right |
|Three fingers moving to the top zone| Go forward|
|Three fingers in the centre zone| Stop|
 
**Notes:**
- Has been trained specifically on right hand being used for all actions except turning left which requires the left hand for more accuracy
- Dropping your hand after turning left or right will allow the car to continue moving forward in its current direction
- Gestures are queued — show 1 finger then 2 fingers while already driving, and the car will go to blue first then red automatically
- Closing your fist will clear the queue and return home
- The car will avoid any obstacles in its direction
 
## Common Issues
 
**Camera not found**
Change `camera_index` from `0` to `1` in `driving.launch.py`.
 
**Model not found error**
You haven't trained the classifier yet. Run `train_classifier.py` first and confirm `src/hand_gesture/models/gesture_classifier.pkl` exists.
 
**`colcon build` fails**
Make sure you sourced ROS2 before building: `source /opt/ros/humble/setup.bash`
 
**GUI shows blank / no video feed**
Wait 10 seconds after launching before opening the browser. PyBullet may take a moment to initialise.
 
**Gestures misclassified**
Retrain the classifier in the same lighting and position you plan to use it. See the confusion matrix after training to identify which gestures are being confused.
```bash
ros2 run hand_gesture collect landmarks
"""
train using the instructions on there
hold up the hand gestures and move around to different zones while saving the samples once complete train classifier again
 
"""
