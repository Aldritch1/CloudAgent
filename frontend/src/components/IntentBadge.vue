<template>
  <div v-if="intent" class="intent-badge">
    <el-tag size="small" type="info">
      意图: {{ intent.intent }} ({{ (intent.confidence * 100).toFixed(0) }}%)
    </el-tag>
  </div>
</template>

<script setup lang="ts">
import type { IntentInfo } from '../types/chat'

defineProps<{ intent?: IntentInfo | null }>()
</script>

<style scoped>
.intent-badge {
  padding: 4px 12px;
  text-align: center;
}
</style>
