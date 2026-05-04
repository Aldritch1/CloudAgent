export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: ToolCall[]
}

export interface ToolCall {
  name: string
  arguments: Record<string, unknown>
  result?: string
}

export interface IntentInfo {
  intent: string
  confidence: number
  target_agent: string
}

export interface SSEEvent {
  event: string
  data: string
}
