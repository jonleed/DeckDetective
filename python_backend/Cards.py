############## Playing Card Detector Functions ###############

# Import necessary packages
import numpy as np
import cv2
import time

### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
### THESE NEED TO MATCH THE SIZE OF THE TRAINING IMAGES. ###
### REFER TO CORNER = WARP[0:CORNER_HEIGHT, 0:CORNER_WIDTH] IN Rank_Suit_Isolator.py ###
CORNER_WIDTH = 50
CORNER_HEIGHT = 100

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 0.15  
SUIT_DIFF_MAX = 0.15


CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.id = None  # Unique identifier for the card
        self.contour = []  # Contour of card
        self.width, self.height = 0, 0  # Width and height of card
        self.corner_pts = []  # Corner points of card
        self.center = []  # Center point of card
        self.warp = []  # 200x300, flattened, grayed, blurred image
        self.rank_img = None  # Thresholded, sized image of card's rank
        self.suit_img = None  # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown"  # Best matched rank
        self.best_suit_match = "Unknown"  # Best matched suit
        self.last_rank = None  # Last known rank
        self.last_suit = None  # Last known suit
        self.rank_diff = 0  # Difference between rank image and best matched train rank image
        self.suit_diff = 0  # Difference between suit image and best matched train suit image

class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"

class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = [] # Thresholded, sized suit image loaded from hard drive
        self.name = "Placeholder"

### Functions ###
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0
    
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
                 'Eight','Nine','Ten','Jack','Queen','King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # Normalize the training rank image
            train_ranks[i].img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        else:
            print(f"Error loading image for rank: {Rank}")
            train_ranks[i].img = None  # Set to None if loading failed

        i += 1

    return train_ranks

def load_suits(filepath):
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""

    train_suits = []
    i = 0
    
    for Suit in ['Spades','Diamonds','Clubs','Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)

        # Normalize the training suit image
        train_suits[i].img = cv2.normalize(train_suits[i].img, None, 0, 255, cv2.NORM_MINMAX)

        i = i + 1

    return train_suits

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    
    return thresh

def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # Find contours and sort their indices by contour size
    cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []

    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) Smaller area than the maximum card size,
    # 2), bigger area than the minimum card size, 3) have no parents,
    # and 4) have four corners

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
                and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts, w, h)

    # Grab corner of warped card image and do a 4x zoom
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1

    # Convert to blur
    Qcorner_blur = cv2.GaussianBlur(Qcorner_zoom, (5, 5), 0)

    # retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)

    # Define a kernel size
    kernel = np.ones((3, 3), np.uint8)

    # Apply morphological opening to remove noise
    query_thresh = cv2.morphologyEx(query_thresh, cv2.MORPH_OPEN, kernel)


    # Split in to top and bottom half (top shows rank, bottom shows suit)
    ### NEEDS TO MATCH THE SIZE OF THE TRAINING IMAGES. ###
    ### REFER TO corner_thresh[20:200, 0:200] IN Rank_Suit_Isolator.py ###
    # Get dimensions of the thresholded corner image
    height, width = query_thresh.shape[:2]

    # Isolate rank from the top part of the corner image
    start_row_rank = int(0.05 * height)
    end_row_rank = int(0.75 * height)
    Qrank = query_thresh[start_row_rank:end_row_rank, 0:width]

    Qsuit = query_thresh[200:400, 0:200]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

        # Normalize rank image
        qCard.rank_img = cv2.normalize(qCard.rank_img, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # print("No rank contours found; setting qCard.rank_img to None")
        qCard.rank_img = None

    # Find suit contour and bounding rectangle, isolate and find largest contour
    Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea,reverse=True)
    
    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

        # Normalize suit image
        qCard.suit_img = cv2.normalize(qCard.suit_img, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # print("No suit contours found; setting qCard.suit_img to None")
        qCard.suit_img = None

    return qCard
  
def match_card(qCard, train_ranks, train_suits):
    """Finds best rank and suit matches for the query card using ORB feature matching for ranks
    and template matching for suits."""

    if qCard.rank_img is None or qCard.rank_img.size == 0:
        # print("Invalid query card rank image")
        return "Unknown", "Unknown", None, None

    if qCard.suit_img is None or qCard.suit_img.size == 0:
        # print("Invalid query card suit image")
        return "Unknown", "Unknown", None, None

    if not train_ranks or not train_suits:
        # print("No training data available")
        return "Unknown", "Unknown", None, None

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

    best_rank_match_name = "Unknown"
    max_rank_matches = 0

    # For rank matching using ORB
    if qCard.rank_img is not None and qCard.rank_img.size != 0:
        # Compute keypoints and descriptors for query rank image
        kp_query_rank, des_query_rank = orb.detectAndCompute(qCard.rank_img, None)

        if des_query_rank is not None:
            for Trank in train_ranks:
                if Trank.img is not None:
                    # Compute keypoints and descriptors for training rank image
                    kp_train_rank, des_train_rank = orb.detectAndCompute(Trank.img, None)

                    # Check if descriptors are not None
                    if des_train_rank is not None:
                        # Match descriptors
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des_query_rank, des_train_rank)

                        # Count the number of good matches
                        num_matches = len(matches)

                        if num_matches > max_rank_matches:
                            max_rank_matches = num_matches
                            best_rank_match_name = Trank.name
    #     else:
    #         print("No descriptors found in query rank image")
    # else:
    #     print("qCard.rank_img is not valid for ORB detection")

    # Set a threshold for minimum number of matches
    MIN_MATCH_COUNT_RANK = 5  # Adjust based on testing
    if max_rank_matches >= MIN_MATCH_COUNT_RANK:
        qCard.best_rank_match = best_rank_match_name
    else:
        qCard.best_rank_match = "Unknown"

    # If ORB matching fails, try template matching for ranks
    if qCard.best_rank_match == "Unknown":
        best_rank_match_diff = float('inf')
        for Trank in train_ranks:
            if Trank.img is not None and qCard.rank_img is not None:
                # Resize query image to match the size of the training image
                query_resized = cv2.resize(qCard.rank_img, (Trank.img.shape[1], Trank.img.shape[0]))
                
                # Ensure both images are the same type
                query_resized = query_resized.astype(np.float32)
                train_img = Trank.img.astype(np.float32)
                
                try:
                    res = cv2.matchTemplate(query_resized, train_img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    rank_diff = 1 - max_val
                    if rank_diff < best_rank_match_diff:
                        best_rank_match_diff = rank_diff
                        qCard.best_rank_match = Trank.name
                except cv2.error as e:
                    print(f"Error in template matching for rank {Trank.name}: {str(e)}")
                    print(f"Query image shape: {query_resized.shape}, Train image shape: {train_img.shape}")
        
        if best_rank_match_diff > RANK_DIFF_MAX:
            qCard.best_rank_match = "Unknown"

    # print(f"Rank matching - Best match: {best_rank_match_name}, Matches: {max_rank_matches}")

    # For suit matching using template matching
    best_suit_match_diff = float('inf')
    best_suit_match_name = "Unknown"

    if qCard.suit_img is not None and qCard.suit_img.size != 0:
        for Tsuit in train_suits:
            if Tsuit.img is not None:
                # Resize query image to match the size of the training image
                query_resized = cv2.resize(qCard.suit_img, (Tsuit.img.shape[1], Tsuit.img.shape[0]))
                
                # Ensure both images are the same type
                query_resized = query_resized.astype(np.float32)
                train_img = Tsuit.img.astype(np.float32)
                
                try:
                    res = cv2.matchTemplate(query_resized, train_img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    suit_diff = 1 - max_val

                    if suit_diff < best_suit_match_diff:
                        best_suit_match_diff = suit_diff
                        best_suit_match_name = Tsuit.name
                except cv2.error as e:
                    print(f"Error in template matching for suit {Tsuit.name}: {str(e)}")
                    print(f"Query image shape: {query_resized.shape}, Train image shape: {train_img.shape}")

        if best_suit_match_diff < SUIT_DIFF_MAX:
            qCard.best_suit_match = best_suit_match_name
            qCard.suit_diff = best_suit_match_diff
        else:
            qCard.best_suit_match = "Unknown"
            qCard.suit_diff = None
    else:
        # print("qCard.suit_img is not valid for template matching")
        qCard.best_suit_match = "Unknown"
        qCard.suit_diff = None

    # Since we are using feature matching for ranks, we don't have rank_diff
    qCard.rank_diff = None

    return qCard.best_rank_match, qCard.best_suit_match, qCard.rank_diff, qCard.suit_diff

def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image, (x,y), 5, (255,0,0), -1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    # Display card name and quality of suit and rank match
    cv2.putText(image,suit_name,(x-60,y+25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,suit_name,(x-60,y+25),font,1,(50,200,200),2,cv2.LINE_AA)

    # Display the card ID for debugging
    cv2.putText(image, f"ID: {qCard.id}", (x - 60, y + 60), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    

    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #r_diff = str(qCard.rank_diff)
    #s_diff = str(qCard.suit_diff)
    #cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    #cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp
