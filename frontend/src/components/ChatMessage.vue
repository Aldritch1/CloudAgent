<template>
  <div :class="['message', message.role]">
    <div class="bubble">{{ message.content }}</div>
  </div>
</template>

<script setup lang="ts">
import type { Message } from '../types/chat'

defineProps<{ message: Message }>()
</script>

<style scoped>
.message {
  display: flex;
  margin: 8px 0;
}
.message.user {
  justify-content: flex-end;
}
.message.assistant {
  justify-content: flex-start;
}
.bubble {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
}
.message.user .bubble {
  background: #409eff;
  color: #fff;
}
.message.assistant .bubble {
  background: #f4f4f5;
  color: #333;
}
</style>
