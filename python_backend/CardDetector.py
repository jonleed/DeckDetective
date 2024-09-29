# CardDetector.py

# Import necessary packages
import cv2
import numpy as np
import os
import Cards
import VideoStream
import time
import math

class CardDetector:
    def __init__(self, videostream, IM_WIDTH=1280, IM_HEIGHT=720, number_of_decks=1):
        ### ---- INITIALIZATION ---- ###
        # Initialize variables for counting
        self.running_count = 0
        self.total_cards_seen = 0
        self.number_of_decks = number_of_decks  # Adjust based on your game setup
        self.total_cards_in_shoe = self.number_of_decks * 52
        self.counted_cards = set()  # Set to store tuples of (Rank, Suit)

        # Initialize last known rank and suit
        self.last_known_rank = None
        self.last_known_suit = None

        # Initialize a list to store cards from the previous frame
        self.previous_cards = []
        self.next_card_id = 1  # Counter to assign unique IDs to cards

        ## Initialize calculated frame rate
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()

        ## Define font to use
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Assign the video stream
        self.videostream = videostream

        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT

        # Load the train rank and suit images
        path = os.path.dirname(os.path.abspath(__file__))
        self.train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
        self.train_suits = Cards.load_suits(path + '/Card_Imgs/')

    ### ---- FUNCTIONS ---- ###
    def match_cards(self, current_cards):
        """
        Match the current detected cards with the previous frame's cards based on proximity.
        Assigns IDs to new cards and updates existing ones.
        """
        for curr_card in current_cards:
            # Default values
            min_distance = float('inf')
            matched_card = None

            # Find the closest previous card to the current card
            for prev_card in self.previous_cards:
                dist = math.hypot(curr_card.center[0] - prev_card.center[0],
                                  curr_card.center[1] - prev_card.center[1])
                if dist < min_distance:
                    min_distance = dist
                    matched_card = prev_card

            # Define a threshold distance to consider cards as the same
            distance_threshold = 50  # Adjust this value based on expected movement between frames

            if matched_card is not None and min_distance < distance_threshold:
                # Assign the same ID and last known rank/suit
                curr_card.id = matched_card.id
                curr_card.last_rank = matched_card.last_rank
                curr_card.last_suit = matched_card.last_suit
            else:
                # Assign a new ID to the card
                curr_card.id = self.next_card_id
                self.next_card_id += 1

    def process_frame(self):
        """
        Processes a single frame from the video stream and updates card counts.
        Returns the processed image and the true count.
        """
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        image = self.videostream.read()

        # Pre-process camera image (gray, blur, and threshold it)
        pre_proc = Cards.preprocess_image(image)

        # Find and sort the contours of all cards in the image (query cards)
        cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

        # Initialize a new "cards" list to assign the card objects.
        current_cards = []
        k = 0

        # If there are contours, process them
        if len(cnts_sort) != 0:

            # For each contour detected:
            for i in range(len(cnts_sort)):
                if cnt_is_card[i] == 1:

                    # Create a card object from the contour
                    card = Cards.preprocess_card(cnts_sort[i], image)

                    # Append to current cards list
                    current_cards.append(card)
                    k += 1

            # Match current cards with previous cards
            self.match_cards(current_cards)

            # For each card, perform matching and update last known values
            for card in current_cards:
                # Find the best rank and suit match for the card.
                rank, suit, rank_diff, suit_diff = Cards.match_card(card, self.train_ranks, self.train_suits)
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

                # Create a tuple of the card's rank and suit
                card_identity = (card.best_rank_match, card.best_suit_match)

                # If this card has not been counted before and both rank and suit are known
                if (card_identity not in self.counted_cards and
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
                    self.running_count += count_value
                    self.total_cards_seen += 1

                    # Add the card to counted cards
                    self.counted_cards.add(card_identity)

                # Draw center point and match result on the image.
                image = Cards.draw_results(image, card)

            # Draw card contours on image
            if len(current_cards) != 0:
                temp_cnts = [card.contour for card in current_cards]
                cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

            # Update previous cards with current cards for the next frame
            self.previous_cards = current_cards

            # Classify cards into player and dealer cards
            player_cards = []
            dealer_cards = []

            for card in current_cards:
                x, y = card.center
                if y < self.IM_HEIGHT / 2:
                    dealer_cards.append(card)
                else:
                    player_cards.append(card)

            # Extract player's hand and dealer's upcard
            player_hand = [card.best_rank_match for card in player_cards if card.best_rank_match != 'Unknown']
            dealer_upcard = dealer_cards[0].best_rank_match if dealer_cards and dealer_cards[0].best_rank_match != 'Unknown' else None

            # Get suggestion
            suggestion = self.get_suggestion(player_hand, dealer_upcard)

            # Draw the suggestion on the image
            if suggestion:
                cv2.putText(image, f"Suggestion: {suggestion}", (10, 100), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            # No cards detected in the current frame
            pass
        
         # Draw the green line to divide dealer and player areas
        cv2.line(image, (0, int(self.IM_HEIGHT / 2)), (self.IM_WIDTH, int(self.IM_HEIGHT / 2)), (0, 255, 0), 2)

        # Display 'Dealer' and 'Player' labels
        cv2.putText(image, "Dealer", (10, int(self.IM_HEIGHT / 4)), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Player", (10, int(3 * self.IM_HEIGHT / 4)), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Calculate number of decks remaining
        number_of_decks_remaining = (self.total_cards_in_shoe - self.total_cards_seen) / 52

        # Avoid division by zero
        if number_of_decks_remaining <= 0:
            true_count = 0
        else:
            true_count = int(self.running_count / number_of_decks_remaining)

        # Optional: Print the true count for debugging
        print(f"True Count: {true_count}")

        # Draw framerate in the corner of the image.
        cv2.putText(image, "FPS: " + str(int(self.frame_rate_calc)), (10, 26), self.font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / self.freq
        self.frame_rate_calc = 1 / time1

        # Return the processed image and the true count
        return image, true_count
    
    def get_suggestion(self, player_hand, dealer_upcard):
        """
        Returns a suggestion ('Hit', 'Stand', 'Double Down', 'Split', 'Surrender')
        based on the player's hand and dealer's upcard.
        """
        # Ensure we have necessary information
        if not player_hand or not dealer_upcard:
            return None

        # Map card ranks to values
        rank_values = {
            'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6,
            'Seven': 7, 'Eight': 8, 'Nine': 9, 'Ten': 10,
            'Jack': 10, 'Queen': 10, 'King': 10, 'Ace': 11
        }

        # Convert player's hand to numerical values
        player_values = [rank_values.get(rank, 0) for rank in player_hand]

        # Handle Aces for soft totals
        total = sum(player_values)
        soft = False
        num_aces = player_values.count(11)

        while total > 21 and num_aces > 0:
            total -= 10
            num_aces -= 1

        if 11 in player_values:
            soft = True

        # Convert dealer's upcard to value
        dealer_value = rank_values.get(dealer_upcard, 0)
        if dealer_value == 11:
            dealer_value = 1  # Treat dealer's Ace as 1

        # Determine suggestion based on basic strategy
        suggestion = self.basic_strategy(total, soft, player_hand, dealer_value)
        return suggestion

    def basic_strategy(self, total, soft, player_hand, dealer_value):
        """
        Determines the suggested action based on basic strategy rules.
        """
        # Surrender rules
        if total == 16 and dealer_value in [9, 10, 1]:
            return 'Surrender'
        if total == 15 and dealer_value == 10:
            return 'Surrender'

        # Splitting logic
        if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
            # Implement splitting rules here if needed
            pass

        # Soft totals
        if soft:
            # Implement soft total rules
            if total == 20:
                return 'Stand'
            elif total == 19:
                if dealer_value == 6:
                    return 'Double Down'
                else:
                    return 'Stand'
            elif total == 18:
                if 2 <= dealer_value <= 6:
                    return 'Double Down'
                elif dealer_value >= 9 or dealer_value == 1:
                    return 'Hit'
                else:
                    return 'Stand'
            elif total == 17:
                if 3 <= dealer_value <= 6:
                    return 'Double Down'
                else:
                    return 'Hit'
            elif total in [13, 14, 15, 16]:
                if 4 <= dealer_value <= 6:
                    return 'Double Down'
                else:
                    return 'Hit'
            else:
                return 'Hit'
        else:
            # Hard totals
            if total >= 17:
                return 'Stand'
            elif 13 <= total <= 16:
                if 2 <= dealer_value <= 6:
                    return 'Stand'
                else:
                    return 'Hit'
            elif total == 12:
                if 4 <= dealer_value <= 6:
                    return 'Stand'
                else:
                    return 'Hit'
            elif total == 11:
                return 'Double Down'
            elif total == 10:
                if 2 <= dealer_value <= 9:
                    return 'Double Down'
                else:
                    return 'Hit'
            elif total == 9:
                if 3 <= dealer_value <= 6:
                    return 'Double Down'
                else:
                    return 'Hit'
            else:
                return 'Hit'


# Add the main block to run the detector individually
if __name__ == "__main__":
    # Camera settings
    IM_WIDTH = 1280
    IM_HEIGHT = 720
    FRAME_RATE = 10

    # Initialize video stream
    videostream = VideoStream.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 2, 0).start()
    time.sleep(1)  # Give the camera time to warm up

    # Initialize card detector
    card_detector = CardDetector(videostream, number_of_decks=1)

    # Main loop
    try:
        while True:
            # Process a frame
            image, true_count = card_detector.process_frame()

            # Display the image
            cv2.imshow("Card Detector", image)

            # Check for 'q' key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Clean up
        cv2.destroyAllWindows()
        videostream.stop()
