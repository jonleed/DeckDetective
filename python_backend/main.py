from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import cv2
import asyncio

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Example: Stream changing numbers
            await websocket.send_text(f"Current Number: {0}")

            # Example: Send video frames (assuming you are capturing from a camera)
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Convert frame to JPEG and then to byte array
            _, jpeg = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(jpeg.tobytes())
    except asyncio.CancelledError:
        print("WebSocket connection cancelled")
    finally:
        await websocket.close()

# This is a helper to open the camera device
camera = cv2.VideoCapture(0)

# Ensure the camera resource is released properly
@app.on_event("shutdown")
async def shutdown_event():
    camera.release()
    print("Camera resource released")