'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { useWebSocket } from './useWebSocket'
import { generationApi, type GenerationRequest, type GenerationResponse } from '@/lib/api'
import toast from 'react-hot-toast'

interface GenerationState {
  isGenerating: boolean
  generationProgress: number
  currentTrack: GenerationResponse | null
  generationHistory: GenerationResponse[]
  error: string | null
}

export function useGeneration() {
  const [state, setState] = useState<GenerationState>({
    isGenerating: false,
    generationProgress: 0,
    currentTrack: null,
    generationHistory: [],
    error: null,
  })

  const { onGenerationUpdate } = useWebSocket()
  const currentGenerationId = useRef<string | null>(null)

  // Listen for WebSocket updates
  useEffect(() => {
    const unsubscribe = onGenerationUpdate((update) => {
      if (update.id === currentGenerationId.current) {
        setState(prev => ({
          ...prev,
          generationProgress: update.progress,
        }))

        if (update.status === 'completed') {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            currentTrack: update,
            generationHistory: [update, ...prev.generationHistory],
          }))
          currentGenerationId.current = null
          toast.success('Music generation completed!')
        } else if (update.status === 'failed') {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            error: update.error || 'Generation failed',
          }))
          currentGenerationId.current = null
          toast.error(update.error || 'Generation failed')
        }
      }
    })

    return unsubscribe
  }, [onGenerationUpdate])

  const generateMusic = useCallback(async (request: GenerationRequest) => {
    try {
      setState(prev => ({
        ...prev,
        isGenerating: true,
        generationProgress: 0,
        error: null,
      }))

      const response = await generationApi.generate(request)
      currentGenerationId.current = response.id

      toast.success('Music generation started!')
      
      // If generation is immediately complete (cached result)
      if (response.status === 'completed') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          currentTrack: response,
          generationHistory: [response, ...prev.generationHistory],
        }))
        currentGenerationId.current = null
      }

    } catch (error: any) {
      setState(prev => ({
        ...prev,
        isGenerating: false,
        error: error.message,
      }))
      toast.error(error.message || 'Failed to start generation')
    }
  }, [])

  const stopGeneration = useCallback(async () => {
    if (currentGenerationId.current) {
      try {
        await generationApi.cancel(currentGenerationId.current)
        setState(prev => ({
          ...prev,
          isGenerating: false,
          generationProgress: 0,
        }))
        currentGenerationId.current = null
        toast.success('Generation cancelled')
      } catch (error: any) {
        toast.error('Failed to cancel generation')
      }
    }
  }, [])

  const saveGeneration = useCallback(async (id: string, metadata?: Record<string, any>) => {
    try {
      await generationApi.save(id, metadata)
      toast.success('Generation saved!')
    } catch (error: any) {
      toast.error('Failed to save generation')
    }
  }, [])

  const loadHistory = useCallback(async () => {
    try {
      const history = await generationApi.getHistory()
      setState(prev => ({
        ...prev,
        generationHistory: history.items,
      }))
    } catch (error: any) {
      console.error('Failed to load generation history:', error)
    }
  }, [])

  // Load history on mount
  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  return {
    ...state,
    generateMusic,
    stopGeneration,
    saveGeneration,
    loadHistory,
  }
}