from fastapi import FastAPI, WebSocket
import cv2
import asyncio
import numpy as np
import os
import time
import Cards
import VideoStream
from CardDetector import CardDetector  # Import the class

app = FastAPI()

# Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

# Initialize video stream
videostream = VideoStream.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 2, 0).start()
time.sleep(1)  # Give the camera time to warm up

# Initialize card detector
card_detector = CardDetector(videostream, IM_WIDTH=1280, IM_HEIGHT=720, number_of_decks=1)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Process a frame
            image, true_count, suggestion = card_detector.process_frame()

            # Send the True Count and suggestion over WebSocket
            await websocket.send_json({"true_count": true_count, "suggestion": suggestion})

            # Convert frame to JPEG and send over WebSocket
            _, jpeg = cv2.imencode('.jpg', image)
            await websocket.send_bytes(jpeg.tobytes())

            # Add a short sleep to yield control
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("WebSocket connection cancelled")
    finally:
        await websocket.close()
        videostream.stop()
        cv2.destroyAllWindows()

