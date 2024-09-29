<template>
    <div v-if="showTutorial" class="tutorial-modal">
      <div class="tutorial-content">
        <h2>Slide {{ currentSlide + 1 }} of {{ slides.length }}</h2>
        <p>{{ slides[currentSlide] }}</p>
  
        <button @click="prevSlide" :disabled="currentSlide === 0">Previous</button>
        <button @click="nextSlide" :disabled="currentSlide === slides.length - 1">Next</button>
        <button @click="closeTutorial">Close</button>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        showTutorial: true,  // Initially set to true to show the tutorial on first load
        currentSlide: 0,     // Current slide index
        slides: [
          "Welcome to the tutorial! Let's walk you through the app.",
          "This is slide 2 with some more information.",
          "Here is slide 3 with additional details.",
          "Slide 4 brings even more useful content.",
          "Congratulations! You've completed the tutorial on slide 5."
        ]
      };
    },
    methods: {
      nextSlide() {
        if (this.currentSlide < this.slides.length - 1) {
          this.currentSlide++;
        }
      },
      prevSlide() {
        if (this.currentSlide > 0) {
          this.currentSlide--;
        }
      },
      closeTutorial() {
        this.showTutorial = false;
      }
    },
    mounted() {
      // Automatically open the tutorial when the component is mounted
      this.showTutorial = true;
    }
  };
  </script>
  
  <style scoped>
  .tutorial-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
  }
  
  .tutorial-content {
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    width: 400px;
  }
  
  button {
    margin: 10px;
  }
  </style>
  