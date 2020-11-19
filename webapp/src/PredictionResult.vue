<template>
  <b-message type="is-cyanprocess" :title="title" has-icon>
    <h2 class="title">{{ data.years }} años, {{ data.months }} meses</h2>
  </b-message>
</template>

<script>
export default {
  props: {
    data: {
      type: Object,
      default: null,
    },
  },
  mounted() {
    this.title += this.data.name;
  },
  data() {
    return {
      title: "Resultado de ",
    };
  },
};
</script>