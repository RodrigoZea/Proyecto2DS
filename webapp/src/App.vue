<template>
  <div id="app" class="container">
    <b-loading is-full-page v-model="isLoading" :can-cancel="false"></b-loading>
    <img alt="tf logo" src="./assets/tf.png" />
    <section class="hero">
      <div class="hero-body">
        <div class="container">
          <h1 class="title">RSNA Edad Ósea</h1>
          <h2 class="subtitle">
            Rodrigo Zea | Gustavo de Leon | Luis Diego Fernandez | Sebastian
            Arriola | Luis Carlos Esturban
          </h2>
        </div>
      </div>
    </section>
    <section class="section">
      <div class="container">
        <b-field class="file is-primary" :class="{ 'has-name': !!chosenImage }">
          <b-upload
            v-model="chosenImage"
            class="file-label"
            @input="onImageChosen"
          >
            <span class="file-cta">
              <b-icon class="file-icon" icon="upload"></b-icon>
              <span class="file-label">Escoger imagen de radiografia</span>
            </span>
            <span class="file-name" v-if="chosenImage">
              {{ chosenImage.name }}
            </span>
          </b-upload>
        </b-field>
      </div>
    </section>
    <section class="section">
      <div v-if="chosenImage">
        <div v-if="currentPredictionYears">
          Predicción: {{ currentPredictionYears }} años,
          {{ currentPredictionMonths }} meses
        </div>
        <div>
          <img ref="img" :src="chosenImage" width="256" height="256" />
        </div>
        <div class="buttons">
          <b-button type="is-primary" @click="modelPredict" :loading="loading"
            >Realizar Prediccion</b-button
          >
        </div>
      </div>
    </section>
  </div>
</template>

<script>
import * as tf from "@tensorflow/tfjs";

export default {
  name: "App",
  data() {
    return {
      model: false,
      chosenImage: null,
      chosenImageData: null,
      currentPredictionYears: null,
      currentPredictionMonths: null,
      isLoading: false,
    };
  },
  mounted() {
    this.loadModel();
  },
  methods: {
    async loadModel() {
      this.model = await tf.loadLayersModel("http://localhost:5555/model.json");
    },
    onImageChosen(value) {
      const fr = new FileReader();
      fr.onload = function () {
        this.$refs.img.src = fr.result;
      }.bind(this);
      fr.readAsDataURL(value);
    },
    preprocessImage(pixelData) {
      const targetDim = 256;
      const edgeSize = 2;
      const resizeDim = targetDim - edgeSize * 2;
      const padVertically = pixelData.width > pixelData.height;
      const padSize = Math.round(
        (Math.max(pixelData.width, pixelData.height) -
          Math.min(pixelData.width, pixelData.height)) /
          2
      );

      const padSquare = padVertically
        ? [
            [padSize, padSize],
            [0, 0],
            [0, 0],
          ]
        : [
            [0, 0],
            [padSize, padSize],
            [0, 0],
          ];

      return tf.tidy(() => {
        // convert the pixel data into a tensor with 1 data channel per pixel
        // i.e. from [h, w, 4] to [h, w, 1]
        let tensor = tf.browser
          .fromPixels(pixelData, 1)
          // pad it until square, such that w = h = max(w, h)
          .pad(padSquare, 255.0);

        // scale it down to smaller than target
        tensor = tf.image
          .resizeBilinear(tensor, [resizeDim, resizeDim])
          // pad it with blank pixels along the edges (to better match the training data)
          .pad(
            [
              [edgeSize, edgeSize],
              [edgeSize, edgeSize],
              [0, 0],
            ],
            255.0
          );

        // normalizar datos y convertir a lo que espera nuestro modelo,
        // los pixeles deben ir en un rango de [-1, 1]
        tensor = tensor.toFloat().div(127.5).sub(tf.scalar(1.0));

        // Reshape again to fit training model [N, 28, 28, 1]
        // where N = 1 in this case
        return tensor.expandDims(0);
      });
    },
    async modelPredict() {
      if (this.chosenImage) {
        // loading
        this.isLoading = true;

        // constantes de nuestro dataset
        const boneage_div = 82.36404279879235;
        const boneage_mean = 127.3207517246848;

        // preprocesar imagen
        let tensor = this.preprocessImage(this.$refs.img, 1);

        // obtener zscore
        const pred_zscore = await this.model.predict(tensor).data();

        // formula para obtener edad en meses
        const pred = boneage_div * pred_zscore[0] + boneage_mean;

        // para mostrar los datos en un formato mas entendible
        this.currentPredictionYears = Math.floor(pred / 12);
        this.currentPredictionMonths = Math.floor(pred % 12);

        this.isLoading = false;
      }
    },
  },
};
</script>

<style lang="scss">
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
