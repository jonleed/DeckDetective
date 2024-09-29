# DeckDetective

**An Interactive Educational Tool for Mastering Blackjack Strategies**

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

DeckDetective is an innovative educational platform designed to help users learn and master blackjack strategies through real-time simulation and feedback. By leveraging computer vision, strategic algorithms, and a user-friendly interface built with **Vue.js**, DeckDetective provides an interactive experience that teaches:

- Basic blackjack strategies
- Card counting techniques
- Strategy deviations based on the true count

Whether you're a beginner looking to understand the fundamentals or an experienced player aiming to refine your skills, DeckDetective offers a hands-on approach to learning blackjack in an engaging and immersive way.

---

## Features

- **Real-Time Card Detection**: Utilizes computer vision to detect and recognize playing cards from a live video stream.
- **Strategy Suggestions**: Provides immediate recommendations based on basic blackjack strategy, including when to hit, stand, double down, split pairs, or surrender.
- **Card Counting Integration**: Tracks the running count and calculates the true count to adjust strategy suggestions dynamically.
- **Strategy Deviations**: Incorporates advanced strategy deviations based on the true count, teaching users when to adjust their play.
- **Interactive Interface**: Displays live video with annotations, including detected cards, true count, and strategic suggestions, all within a Vue.js application.
- **Educational Focus**: Designed to be a learning tool, making complex blackjack strategies accessible and understandable.

---

## How It Works

DeckDetective processes video frames from a webcam or video feed to detect and recognize playing cards on the table. It then:

1. **Detects Cards**: Uses image processing to find cards within the video frame.
2. **Recognizes Cards**: Matches detected cards against a trained dataset to identify the rank and suit.
3. **Calculates Counts**: Updates the running count and true count based on recognized cards using card counting techniques.
4. **Generates Suggestions**: Provides strategy suggestions based on basic blackjack strategy and adjusts them according to the true count.
5. **Displays Information**: Overlays the video feed with annotations showing the detected cards, true count, and strategic advice within a Vue.js frontend.

---

## Prerequisites

Before running DeckDetective, ensure you have the following installed:

- **Python 3.7 or higher**
- **OpenCV**: For image processing tasks.
- **NumPy**: For numerical computations.
- **FastAPI**: For serving the backend application.
- **uvicorn**: For running the ASGI server.
- **WebSocket Support**: For real-time communication between the backend and frontend.
- **Node.js and npm**: For running the frontend application (Vue.js).
- **Vue CLI**: For serving the Vue.js application.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jonleed/DeckDetective.git
cd DeckDetective
```

### 2. Set Up the Python Backend

#### a. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### b. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### c. Install OpenCV (if not installed)

Ensure OpenCV is installed. If not, install it using:

```bash
pip install opencv-python
```

### 3. Set Up the Vue.js Frontend Application

#### a. Navigate to the Frontend Directory

```bash
cd frontend
```

#### b. Install Node.js Dependencies

```bash
npm install
```

---

## Usage

### 1. Start the Backend Server

Navigate back to the root directory (if not already there) and run the FastAPI application:

```bash
uvicorn main:app --reload
```

- The backend server will start on `http://localhost:8000`.

### 2. Start the Frontend Application

In a new terminal, navigate to the `frontend` directory and start the Vue.js application:

```bash
npm run serve
```

- The frontend application will start on `http://localhost:8080` or another available port.

### 3. Access DeckDetective

Open your web browser and navigate to the frontend URL provided in the terminal (e.g., `http://localhost:8080`).

### 4. Using the Application

- **Set Up Your Camera**: Ensure your webcam is connected and positioned to capture the playing area where you'll place the cards.
- **Start Playing**: Place playing cards in view of the camera. The application will detect the cards and display strategy suggestions in real-time.
- **View Suggestions**: The true count and optimal move suggestions will update dynamically as cards are detected and recognized.
- **Interactive Learning**: Use the suggestions to learn about optimal blackjack strategies and how card counting affects decision-making.

---

## Project Structure

- **backend/**: Contains the Python backend application code.
  - `main.py`: FastAPI application with WebSocket endpoint.
  - `CardDetector.py`: Main class for card detection and strategy computation.
  - `Cards.py`: Helper functions and classes for card processing.
  - `VideoStream.py`: Video stream handling.
  - `requirements.txt`: Python dependencies.
- **frontend/**: Contains the Vue.js frontend application code.
  - `src/`: Vue.js components and assets.
    - `App.vue`: Main Vue.js application component.
    - `components/`: Contains Vue.js components like `Socket.vue`.
    - `main.js`: Entry point for the Vue.js application.
  - `package.json`: Node.js dependencies.
- **Card_Imgs/**: Contains the training images for card ranks and suits used for card recognition.
- **README.md**: Project documentation.
- **LICENSE**: License information.

---

## Contributing

Contributions are welcome! If you'd like to contribute to DeckDetective, please follow these steps:

1. **Fork the Repository**: Create your own fork of the project.
2. **Create a Feature Branch**: Develop your feature or fix in a new branch.
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit Your Changes**: Write clear and concise commit messages.
   ```bash
   git commit -m "Add your commit message here"
   ```
4. **Push to Your Fork**: Push your changes to your forked repository.
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Submit a Pull Request**: Describe your changes and submit a pull request for review.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or support, please open an issue on the [GitHub repository](https://github.com/jonleed/DeckDetective) or contact the project maintainers.
---

## Additional Notes

- **Card Dataset**: Ensure you have the training images for card ranks and suits in the `Card_Imgs/` directory. These images are essential for card recognition.
- **Camera Calibration**: For best results, calibrate your camera settings and ensure good lighting conditions to improve card detection accuracy.
- **Error Handling**: The application includes basic error handling, but additional robustness can be added to handle edge cases and improve performance.
- **Extensibility**: DeckDetective is designed to be extensible. Feel free to add new features, such as additional strategy rules, support for different blackjack variations, or enhanced user interfaces.
- **Platform Compatibility**: The application is cross-platform and should run on Windows, macOS, and Linux systems that meet the prerequisites.

---

## Troubleshooting

- **WebSocket Connection Issues**: Ensure that both the backend and frontend are running on the correct ports and that there are no firewall restrictions.
- **Module Not Found Errors**: Verify that all dependencies are installed correctly in your virtual environment.
- **Camera Access Problems**: Check that your webcam is properly connected and that you have granted the necessary permissions for camera access.

---

## Acknowledgements

- **OpenCV**: For providing powerful computer vision tools.
- **Vue.js**: For the progressive JavaScript framework used in the frontend.
- **FastAPI**: For the modern, fast (high-performance) web framework used in the backend.
- **Contributors**: Thanks to everyone who has contributed to this project.

---

Thank you for using DeckDetective! We hope this tool enhances your understanding and enjoyment of blackjack strategies.
