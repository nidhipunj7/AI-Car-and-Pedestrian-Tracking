# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 18:03:09 2020  

@author: Nidhi Punj
"""
import cv2


# Our Image
#video = cv2.VideoCapture("video1.avi")
video = cv2.VideoCapture("PedestriansCompilation.mp4")


# Pre-trained Car classifier
cars_classifier_file = "cars.xml"
pedestrians_classifier_file = "haarcascade_fullbody.xml"

# Create a Car Classifier
car_tracker = cv2.CascadeClassifier(cars_classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrians_classifier_file)


while True:
    
    #Read the current frame
    (read_successful, frame) = video.read()
    
    if read_successful:
        # Convert to grayscale
        grayscaled_frames = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    
        
    # Detect the car
    cars = car_tracker.detectMultiScale(grayscaled_frames)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frames)
    
    # Draw the rectange around the cars
    for (x, y, w, h) in  cars:    
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,255),2) #BGR,thickness-2
        
    # Draw the rectange around the cars
    for (x, y, w, h) in  pedestrians:    
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2) #BGR,thickness-2    

    
    # Display the image with the car spotted
    cv2.imshow("Nidhi's Cars and Pedestrians Detector",frame)
    
    # Don't AutoClose (wait here for a key press)
    cv2.waitKey(1)
    
    print("Yuhooo!! Coding Completed :)")
    
    


"""
# Our Image
img_file = "car_image2.jpg"

# Pre-trained Car classifier
classifier_file = "cars.xml"

# Create the Opencv Image
img  = cv2.imread(img_file)

# Conver it to grayscale (Black and White)
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   

# Create a Car Classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect the car
cars = car_tracker.detectMultiScale(bw)


# Draw the rectange around the cars
for (x, y, w, h) in  cars:    
    cv2.rectangle(img,(x,y),(x+w,y+h), (0,0,255),2) #BGR,thickness-2



# Display the image with the car spotted
cv2.imshow("Nidhi's Cars and Pedestrians Detector",img)



# Don't AutoClose (wait here for a key press)
cv2.waitKey()


print("Yuhooo!! Coding Completed :)")

"""
