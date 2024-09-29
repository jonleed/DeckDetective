############## Python-OpenCV Playing Card Detector ###############

# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards
import VideoStream
import time
import math


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables
# Initialize last known rank and suit
last_known_rank = None
last_known_suit = None

# Initialize a list to store cards from the previous frame
previous_cards = []
next_card_id = 1  # Counter to assign unique IDs to cards

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1) # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')


### ---- FUNCTIONS ---- ###
def match_cards(current_cards, previous_cards):
    """
    Match the current detected cards with the previous frame's cards based on proximity.
    Assigns IDs to new cards and updates existing ones.
    """
    global next_card_id

    for curr_card in current_cards:
        # Default values
        min_distance = float('inf')
        matched_card = None

        # Find the closest previous card to the current card
        for prev_card in previous_cards:
            dist = math.hypot(curr_card.center[0] - prev_card.center[0],
                              curr_card.center[1] - prev_card.center[1])
            if dist < min_distance:
                min_distance = dist
                matched_card = prev_card

        # Define a threshold distance to consider cards as the same
        # Adjust this threshold based on expected movement between frames
        distance_threshold = 50  # You may need to adjust this value

        if matched_card is not None and min_distance < distance_threshold:
            # Assign the same ID and last known rank/suit
            curr_card.id = matched_card.id
            curr_card.last_rank = matched_card.last_rank
            curr_card.last_suit = matched_card.last_suit
        else:
            # Assign a new ID to the card
            curr_card.id = next_card_id
            next_card_id += 1


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0 # Loop control variable

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
	
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # Initialize a new "cards" list to assign the card objects.
    # k indexes the newly made array of cards.
    current_cards = []
    k = 0
    
    # If there are no contours, do nothing
    if len(cnts_sort) != 0:

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):

                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.

                # Create a card object from the contour
                card = Cards.preprocess_card(cnts_sort[i], image)

                # Append to current cards list
                current_cards.append(card)
                k += 1

        # Match current cards with previous cards
        match_cards(current_cards, previous_cards)

        # For each card, perform matching and update last known values
        for card in current_cards:
            # Find the best rank and suit match for the card.
            rank, suit, rank_diff, suit_diff = Cards.match_card(card, train_ranks, train_suits)
            card.best_rank_match = rank
            card.best_suit_match = suit
            card.rank_diff = rank_diff
            card.suit_diff = suit_diff

            # Update last known rank and suit if detection is valid
            if card.best_rank_match != "Unknown":
                card.last_rank = card.best_rank_match
            else:
                if card.last_rank is not None:
                    card.best_rank_match = card.last_rank

            if card.best_suit_match != "Unknown":
                card.last_suit = card.best_suit_match
            else:
                if card.last_suit is not None:
                    card.best_suit_match = card.last_suit

            # Draw center point and match result on the image.
            image = Cards.draw_results(image, card)

        # Draw card contours on image
        if len(current_cards) != 0:
            temp_cnts = [card.contour for card in current_cards]
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

    else:
        # No cards detected in the current frame
        pass  # Optionally, you could handle cards leaving the frame here

    # Update previous cards with current cards for the next frame
    previous_cards = current_cards

    # Draw framerate in the corner of the image.
    cv2.putText(image, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow("Card Detector", image)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Poll the keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

# Clean up
cv2.destroyAllWindows()
videostream.stop()
