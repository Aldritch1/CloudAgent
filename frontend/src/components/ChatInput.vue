<template>
  <div class="chat-input">
    <el-input
      v-model="text"
      :disabled="disabled"
      placeholder="输入消息..."
      @keyup.enter="submit"
    />
    <el-button type="primary" :disabled="disabled || !text.trim()" @click="submit">
      发送
    </el-button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ disabled?: boolean }>()
const emit = defineEmits<{ (e: 'send', text: string): void }>()

const text = ref('')

function submit() {
  if (!text.value.trim() || props.disabled) return
  emit('send', text.value.trim())
  text.value = ''
}
</script>

<style scoped>
.chat-input {
  display: flex;
  gap: 8px;
  padding: 12px;
  border-top: 1px solid #e4e7ed;
}
</style>
