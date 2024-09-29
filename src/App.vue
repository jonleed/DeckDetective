<script setup lang="ts">
import Socket from './components/Socket.vue'
import Tutorial from './components/Tutorial.vue';

const showTutorial = ref(true);  // Controls the visibility of the tutorial
</script>

<template>
  <div class="app-container">
    <h1 class="title">Deck Detective</h1>

    <div class="video-container">
      <!-- Placeholder for the 720p video feed -->
      <Socket />
    </div>

    <div class="values-container">
      <div class="value-item">
        <strong>True Count:</strong>
        <!--<span>{{ trueCount }}</span>-->
      </div>
      <div class="value-item">
        <strong>Optimal Move:</strong>
        <!--<span>{{ optimalMove }}</span>-->
      </div>
    </div>
  </div>
  <HelloWorld msg="Vite + Vue" />
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';

export default defineComponent({
  name: 'DeckDetective',
  setup2() {
    const trueCount = ref(0); // Initialize with your counter logic
    const optimalMove = ref("Stand"); // Replace with your logic for the optimal move

    return { trueCount, optimalMove };
  },
});
</script>

<style scoped>
.app-container {
  text-align: center;
  padding: 20px;
}

.title {
  /* font-size: 2rem; */
  font-size: 4vw;
  margin-top: 0px;
  margin-bottom: 20px;
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
}
</style>
