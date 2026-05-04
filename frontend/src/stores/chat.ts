import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Message, IntentInfo } from '../types/chat'
import { streamChat } from '../api/chat'

function generateId(): string {
  return Math.random().toString(36).substring(2, 10)
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const streaming = ref(false)
  const currentIntent = ref<IntentInfo | null>(null)
  const sessionId = ref(generateId())

  function addMessage(role: Message['role'], content: string) {
    messages.value.push({ id: generateId(), role, content })
  }

  async function sendMessage(content: string) {
    addMessage('user', content)
    streaming.value = true
    currentIntent.value = null

    const assistantMsg: Message = { id: generateId(), role: 'assistant', content: '' }
    messages.value.push(assistantMsg)

    try {
      await streamChat(content, sessionId.value, (event) => {
        if (event.event === 'intent') {
          try {
            currentIntent.value = JSON.parse(event.data)
          } catch {
            // ignore
          }
        } else if (event.event === 'token') {
          assistantMsg.content += event.data
        } else if (event.event === 'tool_call') {
          assistantMsg.content += '\n[调用工具] '
          assistantMsg.content += event.data
        } else if (event.event === 'tool_result') {
          assistantMsg.content += '\n[工具结果] '
          assistantMsg.content += event.data
        } else if (event.event === 'hitl') {
          assistantMsg.content += '\n[需要确认] 请回复"确认"或"取消"。'
        } else if (event.event === 'done') {
          streaming.value = false
        }
      })
    } catch (e) {
      assistantMsg.content += '\n[错误] 连接失败，请重试。'
      streaming.value = false
    }
  }

  return { messages, streaming, currentIntent, sessionId, sendMessage }
})
