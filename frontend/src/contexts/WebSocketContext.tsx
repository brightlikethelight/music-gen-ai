'use client'

import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react'
import { io, Socket } from 'socket.io-client'
import toast from 'react-hot-toast'

interface WebSocketMessage {
  type: string
  data: any
  timestamp: number
}

interface GenerationUpdate {
  id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  stage: string
  audioUrl?: string
  waveformData?: number[]
  error?: string
}

interface WebSocketContextType {
  socket: Socket | null
  isConnected: boolean
  connectionId: string | null
  sendMessage: (type: string, data: any) => void
  onGenerationUpdate: (callback: (update: GenerationUpdate) => void) => () => void
  onUserPresence: (callback: (users: any[]) => void) => () => void
  onChatMessage: (callback: (message: any) => void) => () => void
  joinRoom: (roomId: string) => void
  leaveRoom: (roomId: string) => void
}

const WebSocketContext = createContext<WebSocketContextType | null>(null)

interface WebSocketProviderProps {
  children: React.ReactNode
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [socket, setSocket] = useState<Socket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [connectionId, setConnectionId] = useState<string | null>(null)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  
  const listenersRef = useRef<Map<string, Set<Function>>>(new Map())
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const maxReconnectAttempts = 5
  const reconnectDelay = 1000

  const initializeSocket = useCallback(() => {
    if (socket) {
      socket.disconnect()
    }

    const newSocket = io(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080', {
      transports: ['websocket', 'polling'],
      timeout: 20000,
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: maxReconnectAttempts,
      reconnectionDelay,
      reconnectionDelayMax: 5000,
      forceNew: true,
      auth: {
        // Include authentication token if available
        token: typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null,
      },
    })

    // Connection event handlers
    newSocket.on('connect', () => {
      setIsConnected(true)
      setConnectionId(newSocket.id)
      setReconnectAttempts(0)
      console.log('WebSocket connected:', newSocket.id)
      toast.success('Connected to real-time updates')
    })

    newSocket.on('disconnect', (reason) => {
      setIsConnected(false)
      setConnectionId(null)
      console.log('WebSocket disconnected:', reason)
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, try to reconnect
        handleReconnect()
      }
    })

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error)
      setIsConnected(false)
      handleReconnect()
    })

    newSocket.on('reconnect', (attemptNumber) => {
      console.log('WebSocket reconnected after', attemptNumber, 'attempts')
      toast.success('Reconnected to real-time updates')
    })

    newSocket.on('reconnect_error', (error) => {
      console.error('WebSocket reconnection error:', error)
    })

    newSocket.on('reconnect_failed', () => {
      console.error('WebSocket reconnection failed')
      toast.error('Failed to reconnect to real-time updates')
    })

    // Message handlers
    newSocket.on('generation_update', (data: GenerationUpdate) => {
      notifyListeners('generation_update', data)
    })

    newSocket.on('user_presence', (data: any) => {
      notifyListeners('user_presence', data)
    })

    newSocket.on('chat_message', (data: any) => {
      notifyListeners('chat_message', data)
    })

    newSocket.on('system_notification', (data: any) => {
      console.log('System notification:', data)
      if (data.type === 'info') {
        toast(data.message)
      } else if (data.type === 'warning') {
        toast(data.message, { icon: '⚠️' })
      } else if (data.type === 'error') {
        toast.error(data.message)
      }
    })

    // Ping/pong for connection health
    newSocket.on('ping', () => {
      newSocket.emit('pong')
    })

    setSocket(newSocket)
  }, [socket])

  const handleReconnect = useCallback(() => {
    if (reconnectAttempts >= maxReconnectAttempts) {
      toast.error('Maximum reconnection attempts reached')
      return
    }

    clearTimeout(reconnectTimeoutRef.current)
    reconnectTimeoutRef.current = setTimeout(() => {
      setReconnectAttempts(prev => prev + 1)
      initializeSocket()
    }, reconnectDelay * Math.pow(2, reconnectAttempts))
  }, [reconnectAttempts, initializeSocket])

  const notifyListeners = useCallback((event: string, data: any) => {
    const listeners = listenersRef.current.get(event)
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error('Error in WebSocket listener:', error)
        }
      })
    }
  }, [])

  const sendMessage = useCallback((type: string, data: any) => {
    if (!socket || !isConnected) {
      console.warn('WebSocket not connected, cannot send message')
      return
    }

    const message: WebSocketMessage = {
      type,
      data,
      timestamp: Date.now(),
    }

    socket.emit(type, data)
  }, [socket, isConnected])

  const addEventListener = useCallback((event: string, callback: Function) => {
    if (!listenersRef.current.has(event)) {
      listenersRef.current.set(event, new Set())
    }
    listenersRef.current.get(event)!.add(callback)

    // Return cleanup function
    return () => {
      const listeners = listenersRef.current.get(event)
      if (listeners) {
        listeners.delete(callback)
        if (listeners.size === 0) {
          listenersRef.current.delete(event)
        }
      }
    }
  }, [])

  const onGenerationUpdate = useCallback((callback: (update: GenerationUpdate) => void) => {
    return addEventListener('generation_update', callback)
  }, [addEventListener])

  const onUserPresence = useCallback((callback: (users: any[]) => void) => {
    return addEventListener('user_presence', callback)
  }, [addEventListener])

  const onChatMessage = useCallback((callback: (message: any) => void) => {
    return addEventListener('chat_message', callback)
  }, [addEventListener])

  const joinRoom = useCallback((roomId: string) => {
    if (socket && isConnected) {
      socket.emit('join_room', { roomId })
    }
  }, [socket, isConnected])

  const leaveRoom = useCallback((roomId: string) => {
    if (socket && isConnected) {
      socket.emit('leave_room', { roomId })
    }
  }, [socket, isConnected])

  // Initialize WebSocket connection
  useEffect(() => {
    initializeSocket()

    return () => {
      if (socket) {
        socket.disconnect()
      }
      clearTimeout(reconnectTimeoutRef.current)
    }
  }, [initializeSocket])

  // Cleanup listeners on unmount
  useEffect(() => {
    return () => {
      listenersRef.current.clear()
    }
  }, [])

  // Handle page visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Page is hidden, reduce activity
        if (socket) {
          socket.emit('user_inactive')
        }
      } else {
        // Page is visible, resume activity
        if (socket && !isConnected) {
          initializeSocket()
        } else if (socket) {
          socket.emit('user_active')
        }
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [socket, isConnected, initializeSocket])

  const value: WebSocketContextType = {
    socket,
    isConnected,
    connectionId,
    sendMessage,
    onGenerationUpdate,
    onUserPresence,
    onChatMessage,
    joinRoom,
    leaveRoom,
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

export function useWebSocket() {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}