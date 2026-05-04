<template>
  <div class="sidebar">
    <div class="logo">CloudAgent</div>
    <div class="session-info">
      <p>当前会话</p>
      <p class="session-id">{{ sessionId }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ sessionId: string }>()
</script>

<style scoped>
.sidebar {
  width: 240px;
  background: #1a1a2e;
  color: #fff;
  padding: 16px;
}
.logo {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 24px;
}
.session-id {
  font-size: 12px;
  color: #aaa;
  word-break: break-all;
}
</style>
