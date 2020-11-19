<template>
  <div id="app" class="container">
    <b-loading is-full-page v-model="isLoading" :can-cancel="false"></b-loading>
    <div class="columns">
      <div class="column"><img alt="rsna logo" src="./assets/rsna.png" /></div>
      <div class="column is-narrow"><h1 class="title">+</h1></div>
      <div class="column"><img alt="tf logo" src="./assets/tf.png" /></div>
    </div>
    <section class="hero">
      <div class="hero-body">
        <div class="container">
          <h1 class="title">RSNA EDAD ÓSEA</h1>
          <h2 class="subtitle">
            Rodrigo Zea | Gustavo de León | Luis Diego Fernandez | Sebastian
            Arriola | Luis Carlos Esturbán
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
              <b-radio v-model="selectedModelRadio" native-value="all"
                >Todos</b-radio
              >
            </div>
          </b-step-item>

          <b-step-item step="2" label="Radiografía">
            <div class="container">
              <div v-if="chosenImage">
                <img ref="img" :src="chosenImage" width="256" height="256" />
              </div>
              <div class="buttons" v-if="chosenImage">
                <b-button
                  class="custom-margin"
                  expanded
                  type="is-primary"
                  @click="modelPredict"
                  >Realizar Predicción</b-button
                >
              </div>
              <b-field class="file">
                <b-upload
                  v-model="chosenImage"
                  @input="onImageChosen"
                  expanded
                  :native="true"
                >
                  <a class="button is-primary is-fullwidth custom-margin">
                    <b-icon icon="upload"></b-icon>
                    <span>Escoger Imagen de Radiografía</span>
                  </a>
                </b-upload>
              </b-field>
            </div>
          </b-step-item>

          <b-step-item step="3" label="Resultados">
            <div class="container custom-margin">
              <div
                v-for="i in predictionResults"
                :key="i.name"
                class="result-margin"
              >
                <prediction-result :data="i" />
              </div>
              <div class="container">
                <custom-bar-chart :chart-data="chartData" />
              </div>
              <div class="buttons">
                <b-button
                  class="custom-margin"
                  expanded
                  type="is-primary"
                  @click="reset"
                  >Reiniciar</b-button
                >
              </div>
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

.result-margin {
  margin: {
    top: 8px;
    bottom: 8px;
  }
}
</style>

<script>
import * as tf from "@tensorflow/tfjs";

export default {
  name: "App",
  data() {
    return {
      simpleModel: false,
      convModel: false,
      v3Model: false,
      chosenImage: null,
      predictionResults: [],
      activeStep: 0,
      selectedModelRadio: "simple",
      boneageDiv: 82.36404279879235,
      boneageMean: 127.3207517246848,
      chartData: {},
      isLoading: false,
    };
  },
  mounted() {
    this.loadModels();
  },
  methods: {
    async loadModels() {
      const l = this.$buefy.loading.open();
      this.simpleModel = await tf.loadLayersModel(
        "http://localhost:5555/simple_model/model.json"
      );
      this.convModel = await tf.loadLayersModel(
        "http://localhost:5555/conv_model/model.json"
      );
      this.v3Model = await tf.loadLayersModel(
        "http://localhost:5555/inceptionv3_simple/model.json"
      );
      l.close();
    },
    onImageChosen(value) {
      const fr = new FileReader();
      fr.onload = function () {
        this.$refs.img.src = fr.result;
      }.bind(this);
      fr.readAsDataURL(value);
    },
    generalPreprocessing(pixelData) {
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

        return tensor;
      });
    },
    simpleConvPreprocessing(tensor) {
      // normalizar datos y convertir a lo que espera nuestro modelo,
      // los pixeles deben ir en un rango de [0, 1]
      tensor = tensor.toFloat().div(255);
      return tensor.expandDims(0);
    },
    v3PreprocessImage(tensor) {
      // normalizar datos y convertir a lo que espera nuestro modelo,
      // los pixeles deben ir en un rango de [-1, 1]
      tensor = tensor.toFloat().div(127.5).sub(tf.scalar(1.0));
      return tensor.expandDims(0);
    },
    async simpleModelPredict(tensor) {
      const pred_zscore = await this.simpleModel.predict(tensor).data();

      return this.boneageDiv * pred_zscore[0] + this.boneageMean;
    },
    async convModelPredict(tensor) {
      const pred_zscore = await this.convModel.predict(tensor).data();

      return this.boneageDiv * pred_zscore[0] + this.boneageMean;
    },
    async v3ModelPredict(tensor) {
      // preprocesar imagen
      tensor = this.v3PreprocessImage(tensor);

      // obtener zscore
      const pred_zscore = await this.v3Model.predict(tensor).data();

      // formula para obtener edad en meses
      return this.boneageDiv * pred_zscore[0] + this.boneageMean;
    },
    async modelPredict() {
      // limpiar predicciones previas
      this.predictionResults = [];

      if (this.chosenImage) {
        // preprocesamiento general
        let tensor = this.generalPreprocessing(this.$refs.img, 1);
        let pred;

        if (this.selectedModelRadio == "simple") {
          tensor = this.simpleConvPreprocessing(tensor);
          pred = await this.simpleModelPredict(tensor);

          this.predictionResults.push({
            name: "Red Neuronal Simple",
            years: Math.floor(pred / 12),
            months: Math.floor(pred % 12),
          });

          this.chartData = {
            labels: ["Red Neuronal Simple"],
            datasets: [
              {
                label: "Error Promedio en Meses",
                backgroundColor: "#005DA9",
                data: [55],
                barThickness: 50,
              },
            ],
          };
        } else if (this.selectedModelRadio == "conv") {
          tensor = this.simpleConvPreprocessing(tensor);
          pred = await this.convModelPredict(tensor);

          this.predictionResults.push({
            name: "Red Convolucional",
            years: Math.floor(pred / 12),
            months: Math.floor(pred % 12),
          });

          this.chartData = {
            labels: ["Red Convolucional"],
            datasets: [
              {
                label: "Error Promedio en Meses",
                backgroundColor: "#005DA9",
                data: [16],
                barThickness: 50,
              },
            ],
          };
        } else if (this.selectedModelRadio == "v3") {
          pred = await this.v3ModelPredict(tensor);

          // para mostrar los datos en un formato mas entendible
          this.predictionResults.push({
            name: "InceptionV3",
            years: Math.floor(pred / 12),
            months: Math.floor(pred % 12),
          });

          this.chartData = {
            labels: ["InceptionV3"],
            datasets: [
              {
                label: "Error Promedio en Meses",
                backgroundColor: "#005DA9",
                data: [11],
                barThickness: 50,
              },
            ],
          };
        } else if (this.selectedModelRadio == "all") {
          // simple model
          const tensor1 = this.simpleConvPreprocessing(tensor);
          pred = await this.simpleModelPredict(tensor1);
          this.predictionResults.push({
            name: "Red Neuronal Simple",
            years: Math.floor(pred / 12),
            months: Math.floor(pred % 12),
          });

          // conv model
          pred = await this.convModelPredict(tensor1);
          this.predictionResults.push({
            name: "Red Convolucional",
            years: Math.floor(pred / 12),
            months: Math.floor(pred % 12),
          });

          // v3 model
          pred = await this.v3ModelPredict(tensor);
          // para mostrar los datos en un formato mas entendible
          this.predictionResults.push({
            name: "InceptionV3",
            years: Math.floor(pred / 12),
            months: Math.floor(pred % 12),
          });

          this.chartData = {
            labels: ["Red Neuronal Simple", "Red Convolucional", "InceptionV3"],
            datasets: [
              {
                label: "Error Promedio en Meses",
                backgroundColor: "#005DA9",
                data: [55, 16, 11],
                barThickness: 50,
              },
            ],
          };
        }

        // ir a ultimo paso para mostrar resultado de prediccion
        this.activeStep = 2;
      }
    },
    reset() {
      this.$refs.img.src = null;
      this.chosenImage = null;
      this.predictionResults = [];
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
