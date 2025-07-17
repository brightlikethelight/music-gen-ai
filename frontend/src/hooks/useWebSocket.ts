import { useEffect, useCallback, useRef } from 'react'
import { useWebSocket as useWebSocketContext } from '@/contexts/WebSocketContext'

interface UseWebSocketOptions {
  onGenerationUpdate?: (update: GenerationUpdate) => void
  onCollaborationUpdate?: (update: CollaborationUpdate) => void
  onChatMessage?: (message: ChatMessage) => void
  onUserPresence?: (presence: UserPresence) => void
  onError?: (error: any) => void
  autoReconnect?: boolean
}

interface GenerationUpdate {
  id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  stage: string
  audioUrl?: string
  waveformData?: number[]
  error?: string
  metadata?: Record<string, any>
}

interface CollaborationUpdate {
  user_id: string
  action: string
  data: any
  timestamp: string
}

interface ChatMessage {
  id: string
  user_id: string
  message: string
  timestamp: string
  metadata?: Record<string, any>
}

interface UserPresence {
  user_id: string
  status: 'online' | 'away' | 'offline'
  metadata?: Record<string, any>
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const context = useWebSocketContext()
  const cleanupFunctions = useRef<Array<() => void>>([])

  // Register event listeners
  useEffect(() => {
    if (!context) return

    // Clean up previous listeners
    cleanupFunctions.current.forEach(cleanup => cleanup())
    cleanupFunctions.current = []

    // Generation updates
    if (options.onGenerationUpdate) {
      const cleanup = context.onGenerationUpdate(options.onGenerationUpdate)
      cleanupFunctions.current.push(cleanup)
    }

    // Collaboration updates
    if (options.onCollaborationUpdate) {
      const cleanup = context.onGenerationUpdate((data: any) => {
        if (data.type === 'collaboration_update') {
          options.onCollaborationUpdate?.(data.data)
        }
      })
      cleanupFunctions.current.push(cleanup)
    }

    // Chat messages
    if (options.onChatMessage) {
      const cleanup = context.onChatMessage(options.onChatMessage)
      cleanupFunctions.current.push(cleanup)
    }

    // User presence
    if (options.onUserPresence) {
      const cleanup = context.onUserPresence(options.onUserPresence)
      cleanupFunctions.current.push(cleanup)
    }

    // Cleanup on unmount
    return () => {
      cleanupFunctions.current.forEach(cleanup => cleanup())
      cleanupFunctions.current = []
    }
  }, [
    context,
    options.onGenerationUpdate,
    options.onCollaborationUpdate,
    options.onChatMessage,
    options.onUserPresence,
  ])

  // Helper functions
  const startGeneration = useCallback((taskId: string, prompt: string) => {
    context?.sendMessage('generation_start', {
      task_id: taskId,
      prompt,
    })
  }, [context])

  const cancelGeneration = useCallback((taskId: string) => {
    context?.sendMessage('generation_cancel', {
      task_id: taskId,
    })
  }, [context])

  const sendCollaborationAction = useCallback((
    roomId: string,
    action: string,
    data: any
  ) => {
    context?.sendMessage('collaboration_action', {
      room_id: roomId,
      action,
      data,
    })
  }, [context])

  const sendChatMessage = useCallback((roomId: string, message: string) => {
    context?.sendMessage('chat_message', {
      room_id: roomId,
      message,
    })
  }, [context])

  const updatePresence = useCallback((status: 'online' | 'away' | 'offline') => {
    context?.sendMessage('user_presence', {
      status,
    })
  }, [context])

  const subscribeToChannels = useCallback((channels: string[]) => {
    context?.sendMessage('subscribe', {
      channels,
    })
  }, [context])

  const unsubscribeFromChannels = useCallback((channels: string[]) => {
    context?.sendMessage('unsubscribe', {
      channels,
    })
  }, [context])

  return {
    // Connection state
    isConnected: context?.isConnected ?? false,
    connectionId: context?.connectionId,
    
    // Room management
    joinRoom: context?.joinRoom,
    leaveRoom: context?.leaveRoom,
    
    // Generation
    startGeneration,
    cancelGeneration,
    
    // Collaboration
    sendCollaborationAction,
    
    // Chat
    sendChatMessage,
    
    // Presence
    updatePresence,
    
    // Subscriptions
    subscribeToChannels,
    unsubscribeFromChannels,
    
    // Raw message sending
    sendMessage: context?.sendMessage,
  }
}

// Hook for generation-specific WebSocket functionality
export function useGenerationWebSocket(
  taskId: string | null,
  onUpdate?: (update: GenerationUpdate) => void
) {
  const { startGeneration, cancelGeneration, subscribeToChannels, unsubscribeFromChannels } = useWebSocket({
    onGenerationUpdate: onUpdate,
  })

  // Subscribe to generation updates when taskId changes
  useEffect(() => {
    if (!taskId) return

    // Subscribe to generation-specific channel
    subscribeToChannels([`generation:${taskId}`])

    // Cleanup: unsubscribe when component unmounts or taskId changes
    return () => {
      unsubscribeFromChannels([`generation:${taskId}`])
    }
  }, [taskId, subscribeToChannels, unsubscribeFromChannels])

  return {
    startGeneration: (prompt: string) => {
      if (taskId) {
        startGeneration(taskId, prompt)
      }
    },
    cancelGeneration: () => {
      if (taskId) {
        cancelGeneration(taskId)
      }
    },
  }
}

// Hook for collaboration-specific WebSocket functionality
export function useCollaborationWebSocket(
  roomId: string | null,
  options: {
    onCollaborationUpdate?: (update: CollaborationUpdate) => void
    onChatMessage?: (message: ChatMessage) => void
    onUserPresence?: (presence: UserPresence) => void
  } = {}
) {
  const {
    joinRoom,
    leaveRoom,
    sendCollaborationAction,
    sendChatMessage,
  } = useWebSocket(options)

  // Join/leave room when roomId changes
  useEffect(() => {
    if (!roomId) return

    joinRoom(roomId)

    // Cleanup: leave room when component unmounts or roomId changes
    return () => {
      leaveRoom(roomId)
    }
  }, [roomId, joinRoom, leaveRoom])

  return {
    sendAction: (action: string, data: any) => {
      if (roomId) {
        sendCollaborationAction(roomId, action, data)
      }
    },
    sendMessage: (message: string) => {
      if (roomId) {
        sendChatMessage(roomId, message)
      }
    },
  }
}