import cv2
import numpy as np
import time
import os
import Cards
import VideoStream

### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera.
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1) # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')

def process_card_area(image, train_ranks, train_suits):
    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
    
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    cards = []
    if len(cnts_sort) != 0:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                # Create a card object from the contour and append it to the list of cards.
                card = Cards.preprocess_card(cnts_sort[i], image)
                
                # Find the best rank and suit match for the card.
                card.best_rank_match, card.best_suit_match, card.rank_diff, card.suit_diff = Cards.match_card(card, train_ranks, train_suits)
                
                # Draw center point and match result on the image.
                image = Cards.draw_results(image, card)
                cards.append(card)
    
    # Draw card contours on image
    if cards:
        temp_cnts = [card.contour for card in cards]
        cv2.drawContours(image, temp_cnts, -1, (255,0,0), 2)
    
    return image, cards

### ---- MAIN LOOP ---- ###
cam_quit = 0 # Loop control variable

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Split the image horizontally
    height = image.shape[0]
    width = image.shape[1]
    split_height = height // 2

    dealer_image = image[:split_height, :]
    player_image = image[split_height:, :]

    # Process both areas
    dealer_image, dealer_cards = process_card_area(dealer_image, train_ranks, train_suits)
    player_image, player_cards = process_card_area(player_image, train_ranks, train_suits)

    # Combine the processed images back into one
    combined_image = np.vstack((dealer_image, player_image))

    # Draw a horizontal line to separate dealer and player areas
    cv2.line(combined_image, (0, split_height), (width, split_height), (0, 255, 0), 2)

    # Add labels for dealer and player areas
    cv2.putText(combined_image, "Dealer", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_image, "Player", (10, split_height + 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw framerate in the corner of the image
    cv2.putText(combined_image, f"FPS: {int(frame_rate_calc)}", (width - 120, height - 20), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # Display the image with the identified cards
    cv2.imshow("Card Detector", combined_image)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

# Close all windows and stop the video stream.
cv2.destroyAllWindows()
videostream.stop()