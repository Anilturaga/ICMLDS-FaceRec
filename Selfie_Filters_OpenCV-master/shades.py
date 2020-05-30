from my_CNN_model import *
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the model built in the previous step
my_model = load_my_CNN_model('my_model')

# Face cascade to detect faces
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Define filters-i nvidia
filters = ['images/sunglasses.png','DetectedEdges', 'images/sunglasses_2.png', 'images/sunglasses_3.jpg','GreyScale', 'Cartoon', 'images/sunglasses_5.jpg','filter','images/sunglasses_6.png']
filterIndex = 0

# Load the video
camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 920)
camera.set(cv2.CAP_PROP_FPS, 7)
# Keep looping
while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Add the 'Next Filter' button to the frame
    frame = cv2.rectangle(frame, (500,10), (620,65), (235,50,50), -1)
    cv2.putText(frame, "NEXT FILTER", (512, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Find contours (bottle cap in my case) in the image
    (cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check to see if any contours were found
    if len(cnts) > 0:
    	# Sort the contours and find the largest one -- we
    	# will assume this contour correspondes to the area of the bottle cap
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 65:
            if 500 <= center[0] <= 620: # Next Filter
                filterIndex += 1
                filterIndex %= 9
                continue

    for (x, y, w, h) in faces:
    	# Grab the face
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        # Normalize to match the input format of the model - Range of pixel to [0, 1]
        gray_normalized = gray_face / 255

        # Resize it to 96x96 to match the input format of the model
        original_shape = gray_face.shape # A Copy for future reference
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_copy = face_resized.copy()
        face_resized = face_resized.reshape(1, 96, 96, 1)

        # Predicting the keypoints using the model
        keypoints = my_model.predict(face_resized)

        # De-Normalize the keypoints values
        keypoints = keypoints * 48 + 48

        # Map the Keypoints back to the original image
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_color2 = np.copy(face_resized_color)

        # Pair them together
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))
        if filterIndex == 1:
            frame = cv2.Canny(frame,100,200)
            cv2.imshow("Frame", frame)
        elif filterIndex == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Frame", frame)
        elif filterIndex == 5:
            """img_color = frame
            num_down = 2       # number of downsampling steps
            num_bilateral = 7  # number of bilateral filtering steps
            for x in range(num_down):
                img_color = cv2.pyrDown(img_color)
                # repeatedly apply small bilateral filter instead of
                # applying one large filter
            for x in range(num_bilateral):
                img_color = cv2.bilateralFilter(img_color, d=9,sigmaColor=9,sigmaSpace=7)
            # upsample image to original size
            for x in range(num_down):
                img_color = cv2.pyrUp(img_color)
            # convert to grayscale and apply median blur
            img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            img_blur = cv2.medianBlur(img_gray, 7)
            # detect and enhance edges
            img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=9,C=2)
            # convert back to color, bit-AND with color image
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
            img_cartoon = cv2.bitwise_and(img_color, img_edge)
            # display
            cv2.imshow("Frame", img_cartoon)"""
        elif filterIndex == 7:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow("Frame", frame)      
        else:
            # Add FILTER to the frame
            sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
            sunglass_width = int((points[7][0]-points[9][0])*1.1)
            sunglass_height = int((points[10][1]-points[8][1])/1.1)
            sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
            transparent_region = sunglass_resized[:,:,:3] != 0
            face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]

        	# Resize the face_resized_color image back to its original shape
            frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)
            cv2.imshow("Frame",frame)

        # Add KEYPOINTS to the frame2
        """for keypoint in points:
            cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)

        frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)

        # Show the frame and the frame2
        cv2.imshow("Selfie Filters", frame)
        if filterIndex == 1:
            edge_img = cv2.Canny(frame,100,200)
            cv2.imshow("Detected Edges", edge_img)
        if filterIndex == 2:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Detected Edges", gray_img)
        if filterIndex == 3:
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow("Detected Edges", hsv_img)
       
        
        cv2.imshow("Facial Keypoints", frame2)
"""
    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
