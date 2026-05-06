import type { SSEEvent } from '../types/chat'

export async function streamChat(
  message: string,
  sessionId: string,
  onEvent: (event: SSEEvent) => void,
): Promise<void> {
  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }

  const reader = response.body?.getReader()
  const decoder = new TextDecoder()
  if (!reader) return

  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    let currentEvent = ''
    let currentData = ''

    for (const rawLine of lines) {
      const line = rawLine.trim()
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7)
      } else if (line.startsWith('data: ')) {
        const dataLine = line.slice(6)
        currentData = currentData ? currentData + '\n' + dataLine : dataLine
      } else if (line === '' && currentEvent) {
        onEvent({ event: currentEvent, data: currentData })
        currentEvent = ''
        currentData = ''
      }
    }
  }
}
