import Vue from 'vue';
import App from './App.vue';
import Buefy from 'buefy';
import './scss/styles.scss';

Vue.use(Buefy)
Vue.config.productionTip = false;
Vue.component('custom-bar-chart', require('./BarChart.vue').default);

new Vue({
  render: h => h(App),
}).$mount('#app');
