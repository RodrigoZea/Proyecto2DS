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
      <div class="block">
        <b-steps v-model="activeStep" animated rounded has-navigation>
          <b-step-item step="1" label="Modelo">
            <h1 class="title has-text-centered">Selecciona un Modelo</h1>
            <div class="block">
              <b-radio v-model="selectedModelRadio" native-value="simple"
                >Simple</b-radio
              >
              <b-radio v-model="selectedModelRadio" native-value="conv"
                >Red Convolucional</b-radio
              >
              <b-radio v-model="selectedModelRadio" native-value="v3"
                >InceptionV3</b-radio
              >
            </div>
          </b-step-item>

          <b-step-item step="2" label="Radiografia">
            <div class="container">
              <div v-if="chosenImage.name">
                <img ref="img" :src="chosenImage" width="256" height="256" />
              </div>
              <div class="buttons" v-if="chosenImage.name">
                <b-button
                  class="custom-margin"
                  expanded
                  type="is-success"
                  @click="modelPredict"
                  >Realizar Prediccion</b-button
                >
              </div>
              <b-field class="file">
                <b-upload v-model="chosenImage" @input="onImageChosen" expanded>
                  <a class="button is-primary is-fullwidth custom-margin">
                    <b-icon icon="upload"></b-icon>
                    <span>Escoger Imagen de Radiografía</span>
                  </a>
                </b-upload>
              </b-field>
            </div>
          </b-step-item>

          <b-step-item step="3" label="Predicción">
            <div class="container custom-margin">
              <b-message
                type="is-success"
                title="Resultado de Predicción"
                has-icon
              >
                <h2 class="title">
                  {{ currentPredictionYears }} años,
                  {{ currentPredictionMonths }} meses
                </h2>
                <div class="buttons">
                  <b-button
                    class="custom-margin"
                    expanded
                    type="is-success"
                    @click="reset"
                    >Reiniciar</b-button
                  >
                </div>
              </b-message>
            </div>
          </b-step-item>
        </b-steps>
      </div>
    </section>
  </div>
</template>

<style lang="scss" scoped>
.custom-margin {
  margin: {
    left: 25%;
    right: 25%;
  }
}
</style>

<script>
import * as tf from "@tensorflow/tfjs";

export default {
  name: "App",
  data() {
    return {
      model: false,
      chosenImage: {},
      chosenImageData: null,
      currentPredictionYears: null,
      currentPredictionMonths: null,
      isLoading: false,
      activeStep: 0,
      selectedModelRadio: "simple",
    };
  },
  mounted() {
    this.loadModel();
  },
  methods: {
    async loadModel() {
      this.model = await tf.loadLayersModel(
        "http://localhost:5555/inceptionv3_simple/model.json"
      );
    },
    onImageChosen(value) {
      const fr = new FileReader();
      fr.onload = function () {
        this.$refs.img.src = fr.result;
      }.bind(this);
      fr.readAsDataURL(value);
    },
    v3PreprocessImage(pixelData) {
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
        tensor = tf.image.resizeBilinear(tensor, [resizeDim, resizeDim]).pad(
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

        return tensor.expandDims(0);
      });
    },
    async v3ModelPredict() {
      // constantes de nuestro dataset
      const boneage_div = 82.36404279879235;
      const boneage_mean = 127.3207517246848;

      // preprocesar imagen
      let tensor = this.v3PreprocessImage(this.$refs.img, 1);

      // obtener zscore
      const pred_zscore = await this.model.predict(tensor).data();

      // formula para obtener edad en meses
      return boneage_div * pred_zscore[0] + boneage_mean;
    },
    async modelPredict() {
      if (this.chosenImage) {
        // loading on
        const l = this.$buefy.loading.open();
        let pred;

        if (this.selectedModelRadio == "simple") {
          return;
        } else if (this.selectedModelRadio == "conv") {
          return;
        } else if (this.selectedModelRadio == "v3") {
          pred = await this.v3ModelPredict();
        }

        // para mostrar los datos en un formato mas entendible
        this.currentPredictionYears = Math.floor(pred / 12);
        this.currentPredictionMonths = Math.floor(pred % 12);

        // ir a ultimo paso para mostrar resultado de prediccion
        this.activeStep = 2;

        // loading off
        l.close();
      }
    },
    reset() {
      this.chosenImage = {};
      this.currentPredictionYears = null;
      this.currentPredictionMonth = null;
      this.activeStep = 0;
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
