from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import cv2
import asyncio
import numpy as np
import os
import time
import math
import Cards
import VideoStream

app = FastAPI()

# Initialize variables for counting
running_count = 0
total_cards_seen = 0
number_of_decks = 1  # Adjust based on your game setup
total_cards_in_shoe = number_of_decks * 52
counted_cards = set()  # Set to store tuples of (Rank, Suit)
previous_cards = []
next_card_id = 1  # Counter to assign unique IDs to cards

# Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

# Initialize font
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize video stream
videostream = VideoStream.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 2, 0).start()
time.sleep(1)  # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global running_count, total_cards_seen, counted_cards, previous_cards, next_card_id
    await websocket.accept()
    try:
        while True:
            # Start time for frame rate calculation
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            image = videostream.read()

            # Pre-process camera image
            pre_proc = Cards.preprocess_image(image)

            # Find and sort the contours of all cards in the image
            cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

            current_cards = []
            k = 0

            if len(cnts_sort) != 0:
                for i in range(len(cnts_sort)):
                    if cnt_is_card[i] == 1:
                        card = Cards.preprocess_card(cnts_sort[i], image)
                        current_cards.append(card)
                        k += 1

                match_cards(current_cards, previous_cards)

                for card in current_cards:
                    rank, suit, rank_diff, suit_diff = Cards.match_card(card, train_ranks, train_suits)
                    card.best_rank_match = rank
                    card.best_suit_match = suit
                    card.rank_diff = rank_diff
                    card.suit_diff = suit_diff

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

                    # Create a tuple of the card's rank and suit
                    card_identity = (card.best_rank_match, card.best_suit_match)

                    # If this card has not been counted before and both rank and suit are known
                    if (card_identity not in counted_cards and
                            card.best_rank_match != "Unknown" and
                            card.best_suit_match != "Unknown"):
                        # Assign count value based on rank
                        rank_name = card.best_rank_match
                        if rank_name in ['Two', 'Three', 'Four', 'Five', 'Six']:
                            count_value = 1
                        elif rank_name in ['Ten', 'Jack', 'Queen', 'King', 'Ace']:
                            count_value = -1
                        else:
                            count_value = 0  # For Seven, Eight, Nine

                        # Update running count and total cards seen
                        running_count += count_value
                        total_cards_seen += 1

                        # Add the card to counted cards
                        counted_cards.add(card_identity)

                    # Draw results on image
                    image = Cards.draw_results(image, card)

                # Draw card contours on image
                if len(current_cards) != 0:
                    temp_cnts = [card.contour for card in current_cards]
                    cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

                # Update previous cards
                previous_cards[:] = current_cards

            # Calculate number of decks remaining
            number_of_decks_remaining = (total_cards_in_shoe - total_cards_seen) / 52

            # Avoid division by zero
            if number_of_decks_remaining <= 0:
                true_count = 0
            else:
                true_count = running_count / number_of_decks_remaining

            # Send the True Count over WebSocket
            await websocket.send_json({"true_count": true_count})

            # Convert frame to JPEG and then to byte array
            _, jpeg = cv2.imencode('.jpg', image)
            await websocket.send_bytes(jpeg.tobytes())

            # Frame rate calculation
            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / cv2.getTickFrequency()
            frame_rate_calc = 1 / time1

            # Add a short sleep to yield control
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("WebSocket connection cancelled")
    finally:
        await websocket.close()
        # Release resources
        videostream.stop()
        cv2.destroyAllWindows()

# This is a helper to open the camera device
# camera = cv2.VideoCapture(0)

# Ensure the camera resource is released properly
@app.on_event("shutdown")
async def shutdown_event():
    camera.release()
    print("Camera resource released")

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
        distance_threshold = 50  # Adjust based on expected movement

        if matched_card is not None and min_distance < distance_threshold:
            # Assign the same ID and last known rank/suit
            curr_card.id = matched_card.id
            curr_card.last_rank = matched_card.last_rank
            curr_card.last_suit = matched_card.last_suit
        else:
            # Assign a new ID to the card
            curr_card.id = next_card_id
            next_card_id += 1