<template>
  <div>
    <!-- <h1>{{ liveNumber }}</h1> -->
    
    <div class="video-container">
      <!-- Placeholder for the 720p video feed -->
      <img :src="videoFrame"  id="cameraFeed" />
    </div>

    <div class="values-container">
      <div class="value-item">
        <strong>True Count:</strong>
        <div>
          <span>{{ trueCount }}</span>
        </div>
      </div>
      <div class="value-item">
        <strong>Optimal Move:</strong>
        <div>
          <span>{{ suggestion }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      suggestion: '',
      trueCount: 0,
      videoFrame: null
    };
  },
  mounted() {
    this.setupWebSocket();
  },
  methods: {
    async setupWebSocket() {
      const socket = new WebSocket("ws://localhost:8000/ws");
      socket.binaryType = 'arraybuffer';  // To handle video frames

      await new Promise((resolve) => {
        socket.onopen = resolve;
      });

      socket.onmessage = (event) => {
        if (typeof event.data === 'string') {
          let jsonData = ''
          try {
            jsonData = JSON.parse(event.data);
            this.suggestion = jsonData.suggestion;
            this.trueCount = jsonData.true_count;
          } catch {
            return
          }
        } else if (event.data instanceof ArrayBuffer) {
          // Handle binary message (e.g., video frame)
          const blob = new Blob([event.data], { type: 'image/jpeg' });
          this.videoFrame = URL.createObjectURL(blob);
        }
      };
    }
  }
};
</script>

<style>
#cameraFeed {
  max-width: 90%;
  height: auto;
}

.app-container {
  text-align: center;
  padding: 20px;
}

.title {
  /* font-size: 2rem; */
  font-size: 4vw;
  margin-top: 0px;
  margin-bottom: 20px;
  color: white;
}

.video-container {
  margin-bottom: 20px;
}

.video-feed {
  max-width: 100%;
  height: auto;
  border: 2px solid #ccc; /* Optional: Add a border for styling */
}

.values-container {
  display: flex;
  justify-content: center;
  gap: 40px; /* Space between the two value items */
}

.value-item {
  font-size: 4vh; /* Adjust the size as needed */
  color: white;
}
</style>
