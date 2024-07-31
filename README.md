# Driver-drowsiness-detection-using-Visual-behavior-
Engineered a comprehensive road safety system utilizing TensorFlow and OpenCV for real-time facial monitoring, integrated with Telegram API and Geolocator for timely driver drowsiness alerts.

We first started with Haar cascades and mtcnn but we weren’t able to decrease the false positive rate of detecting faces. Later we deploying dlib library into the system. By this method, firstly if the driver is observed as drowsy, the change on the interface is first seen and then an alert is played, this continues to play as long as the driver is drowsy along with that after certain consecutive frames, if the driver is still drowsy a message is sent to the user with the current location to ensure the safety of the diver.
