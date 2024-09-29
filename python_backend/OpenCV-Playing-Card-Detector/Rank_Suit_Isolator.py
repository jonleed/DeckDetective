### Takes a card picture and creates a top-down 200x300 flattened image
### of it. Isolates the suit and rank and saves the isolated images.
### Runs through A - K ranks and then the 4 suits.

# Import necessary packages
import cv2
import numpy as np
import time
import Cards
import os

img_path = os.path.dirname(os.path.abspath(__file__)) + '/Card_Imgs/'

IM_WIDTH = 1280
IM_HEIGHT = 720

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# If using a USB Camera instead of a PiCamera, change PiOrUSB to 2
PiOrUSB = 2

if PiOrUSB == 1:
    # Import packages from picamera library
    from picamera.array import PiRGBArray
    from picamera import PiCamera

    # Initialize PiCamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))

if PiOrUSB == 2:
    # Initialize USB camera
    cap = cv2.VideoCapture(0)

# Use counter variable to switch from isolating Rank to isolating Suit
i = 1

for Name in ['Ace','Two','Three','Four','Five','Six','Seven','Eight',
             'Nine','Ten','Jack','Queen','King','Spades','Diamonds',
             'Clubs','Hearts']:

    filename = Name + '.jpg'
    capture_successful = False  # Flag to indicate successful capture

    while not capture_successful:
        print('Press "p" to take a picture of ' + filename)

        if PiOrUSB == 1: # PiCamera
            rawCapture.truncate(0)
            # Press 'p' to take a picture
            for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

                image = frame.array
                cv2.imshow("Card",image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    break

                rawCapture.truncate(0)

        if PiOrUSB == 2: # USB camera
            # Press 'p' to take a picture
            while(True):

                ret, frame = cap.read()
                cv2.imshow("Card",frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    image = frame
                    break

        # Pre-process image
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        retval, thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)

        # Find contours and sort them by size
        cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea,reverse=True)

        # Assume largest contour is the card. If there are no contours, print an error
        flag = 0
        image2 = image.copy()

        if len(cnts) == 0:
            print('No contours found!')
            print('Press "r" to retry, or "q" to quit.')
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                continue  # Retry capturing the image
            elif key == ord('q'):
                cv2.destroyAllWindows()
                if PiOrUSB == 2:
                    cap.release()
                if PiOrUSB == 1:
                    camera.close()
                exit()
            else:
                continue

        card = cnts[0]

        # Approximate the corner points of the card
        peri = cv2.arcLength(card,True)
        approx = cv2.approxPolyDP(card,0.01*peri,True)
        pts = np.float32(approx)

        x,y,w,h = cv2.boundingRect(card)

        # Flatten the card and convert it to 200x300
        warp = Cards.flattener(image,pts,w,h)

        # Increase the width and height to capture more of the corner
        corner = warp[0:100, 0:50]

        #corner_gray = cv2.cvtColor(corner,cv2.COLOR_BGR2GRAY)
        corner_zoom = cv2.resize(corner, (0,0), fx=4, fy=4)
        corner_blur = cv2.GaussianBlur(corner_zoom,(5, 5),0)
        retval, corner_thresh = cv2.threshold(corner_blur, 155, 255, cv2. THRESH_BINARY_INV)

        # Apply morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        corner_thresh = cv2.morphologyEx(corner_thresh, cv2.MORPH_OPEN, kernel)

        # Visualize the thresholded corner
        cv2.imshow("Corner Threshold", corner_thresh)
        # cv2.waitKey(0)

        # Isolate suit or rank
        if i <= 13: # Isolate rank
            height, width = corner_thresh.shape[:2]
            start_row = int(0.05 * height)
            end_row = int(0.75 * height)
            rank = corner_thresh[start_row:end_row, 0:width]
            cv2.imshow("Rank Image", rank)
            # cv2.waitKey(0)
            rank_cnts, hier = cv2.findContours(rank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rank_cnts = sorted(rank_cnts, key=cv2.contourArea, reverse=True)
            if len(rank_cnts) > 0:
                x, y, w, h = cv2.boundingRect(rank_cnts[0])
                rank_roi = rank[y:y+h, x:x+w]
                rank_sized = cv2.resize(rank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
                final_img = rank_sized
                # Normalize before saving
                final_img = cv2.normalize(final_img, None, 0, 255, cv2.NORM_MINMAX)
            else:
                print(f"No contours found in rank image for {filename}!")
                print('Press "r" to retry, or "q" to quit.')
                key = cv2.waitKey(0) & 0xFF
                if key == ord('r'):
                    continue  # Retry capturing the image
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    if PiOrUSB == 2:
                        cap.release()
                    if PiOrUSB == 1:
                        camera.close()
                    exit()
                else:
                    continue

        if i > 13: # Isolate suit
            suit = corner_thresh[200:400, 0:200] # Adjust these indices if necessary
            cv2.imshow("Suit Image", suit)
            # cv2.waitKey(0)
            suit_cnts, hier = cv2.findContours(suit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            suit_cnts = sorted(suit_cnts, key=cv2.contourArea, reverse=True)
            if len(suit_cnts) > 0:
                x, y, w, h = cv2.boundingRect(suit_cnts[0])
                suit_roi = suit[y:y+h, x:x+w]
                suit_sized = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
                final_img = suit_sized
                # Normalize before saving
                final_img = cv2.normalize(final_img, None, 0, 255, cv2.NORM_MINMAX)
            else:
                print(f"No contours found in rank image for {filename}!")
                print('Press "r" to retry, or "q" to quit.')
                key = cv2.waitKey(0) & 0xFF
                if key == ord('r'):
                    continue  # Retry capturing the image
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    if PiOrUSB == 2:
                        cap.release()
                    if PiOrUSB == 1:
                        camera.close()
                    exit()
                else:
                    continue


        cv2.imshow("Final Image",final_img)

        # Save image, retry or quit
        print('Press "c" to continue and save the image, "r" to retry, or "q" to quit.')
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):
            # Save image
            cv2.imwrite(img_path + filename, final_img)
            capture_successful = True  # Exit the while loop
        elif key == ord('r'):
            # Retry: Go back to the start of the while loop
            continue
        elif key == ord('q'):
            cv2.destroyAllWindows()
            if PiOrUSB == 2:
                cap.release()
            if PiOrUSB == 1:
                camera.close()
            exit()
        else:
            # Any other key, default to retry
            continue

    i = i + 1

cv2.destroyAllWindows()

# Release cap for USB cam
if PiOrUSB == 2:
    cap.release()

if PiOrUSB == 1:
    camera.close()