<template>
  <div class="chat-container">
    <Sidebar :session-id="store.sessionId" />
    <div class="chat-main">
      <div class="messages" ref="messagesRef">
        <ChatMessage
          v-for="msg in store.messages"
          :key="msg.id"
          :message="msg"
        />
      </div>
      <IntentBadge :intent="store.currentIntent" />
      <ChatInput
        :disabled="store.streaming"
        @send="store.sendMessage"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { useChatStore } from '../stores/chat'
import Sidebar from '../components/Sidebar.vue'
import ChatMessage from '../components/ChatMessage.vue'
import ChatInput from '../components/ChatInput.vue'
import IntentBadge from '../components/IntentBadge.vue'

const store = useChatStore()
const messagesRef = ref<HTMLDivElement>()

watch(() => store.messages.length, () => {
  nextTick(() => {
    messagesRef.value?.scrollTo({ top: messagesRef.value.scrollHeight, behavior: 'smooth' })
  })
})
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
}
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #fff;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
</style>
