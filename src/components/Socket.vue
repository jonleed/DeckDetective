<template>
  <div>
    <h1>{{ liveNumber }}</h1>
    <img :src="videoFrame" />
  </div>
</template>

<script>
export default {
  data() {
    return {
      liveNumber: null,
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
          // Handle text message (e.g., changing numbers)
          this.liveNumber = event.data;
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
