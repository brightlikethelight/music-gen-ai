'use client'

import { useState, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'

export interface Track {
  id: string
  name: string
  type: 'audio' | 'midi'
  audioUrl?: string
  muted: boolean
  solo: boolean
  volume: number
  pan: number
  effects: Effect[]
  waveformData: number[]
  color: string
  duration: number
  startTime: number
}

export interface Effect {
  id: string
  type: string
  name: string
  enabled: boolean
  parameters: Record<string, any>
}

export interface EditorAction {
  type: string
  payload: any
  timestamp: number
}

export interface AudioEditorState {
  tracks: Track[]
  selectedTrack: string | null
  history: EditorAction[]
  historyIndex: number
  clipboard: any | null
}

const TRACK_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
  '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
]

export function useAudioEditor() {
  const [state, setState] = useState<AudioEditorState>({
    tracks: [],
    selectedTrack: null,
    history: [],
    historyIndex: -1,
    clipboard: null,
  })

  const actionId = useRef(0)

  // History management
  const addAction = useCallback((action: Omit<EditorAction, 'timestamp'>) => {
    setState(prev => {
      const newAction: EditorAction = {
        ...action,
        timestamp: Date.now(),
      }
      
      // Remove any actions after current index (for redo)
      const newHistory = prev.history.slice(0, prev.historyIndex + 1)
      newHistory.push(newAction)
      
      // Limit history size
      if (newHistory.length > 100) {
        newHistory.shift()
      }
      
      return {
        ...prev,
        history: newHistory,
        historyIndex: newHistory.length - 1,
      }
    })
  }, [])

  // Track operations
  const addTrack = useCallback((trackData?: Partial<Track>) => {
    const newTrack: Track = {
      id: uuidv4(),
      name: `Track ${state.tracks.length + 1}`,
      type: 'audio',
      muted: false,
      solo: false,
      volume: 0.8,
      pan: 0,
      effects: [],
      waveformData: [],
      color: TRACK_COLORS[state.tracks.length % TRACK_COLORS.length],
      duration: 0,
      startTime: 0,
      ...trackData,
    }

    setState(prev => ({
      ...prev,
      tracks: [...prev.tracks, newTrack],
      selectedTrack: newTrack.id,
    }))

    addAction({
      type: 'ADD_TRACK',
      payload: { track: newTrack },
    })
  }, [state.tracks.length, addAction])

  const removeTrack = useCallback((trackId: string) => {
    setState(prev => {
      const trackToRemove = prev.tracks.find(t => t.id === trackId)
      if (!trackToRemove) return prev

      const newTracks = prev.tracks.filter(t => t.id !== trackId)
      const newSelectedTrack = prev.selectedTrack === trackId 
        ? (newTracks.length > 0 ? newTracks[0].id : null)
        : prev.selectedTrack

      return {
        ...prev,
        tracks: newTracks,
        selectedTrack: newSelectedTrack,
      }
    })

    addAction({
      type: 'REMOVE_TRACK',
      payload: { trackId },
    })
  }, [addAction])

  const updateTrack = useCallback((trackId: string, updates: Partial<Track>) => {
    setState(prev => ({
      ...prev,
      tracks: prev.tracks.map(track =>
        track.id === trackId ? { ...track, ...updates } : track
      ),
    }))

    addAction({
      type: 'UPDATE_TRACK',
      payload: { trackId, updates },
    })
  }, [addAction])

  const selectTrack = useCallback((trackId: string | null) => {
    setState(prev => ({
      ...prev,
      selectedTrack: trackId,
    }))
  }, [])

  // Edit operations
  const cutSelection = useCallback(() => {
    const selectedTrack = state.tracks.find(t => t.id === state.selectedTrack)
    if (!selectedTrack) return

    // Implement cut logic here
    setState(prev => ({
      ...prev,
      clipboard: {
        type: 'cut',
        data: selectedTrack,
        timestamp: Date.now(),
      },
    }))

    addAction({
      type: 'CUT_SELECTION',
      payload: { trackId: selectedTrack.id },
    })
  }, [state.tracks, state.selectedTrack, addAction])

  const copySelection = useCallback(() => {
    const selectedTrack = state.tracks.find(t => t.id === state.selectedTrack)
    if (!selectedTrack) return

    setState(prev => ({
      ...prev,
      clipboard: {
        type: 'copy',
        data: selectedTrack,
        timestamp: Date.now(),
      },
    }))

    addAction({
      type: 'COPY_SELECTION',
      payload: { trackId: selectedTrack.id },
    })
  }, [state.tracks, state.selectedTrack, addAction])

  const pasteSelection = useCallback(() => {
    if (!state.clipboard) return

    if (state.clipboard.type === 'copy') {
      const newTrack = {
        ...state.clipboard.data,
        id: uuidv4(),
        name: `${state.clipboard.data.name} (Copy)`,
      }
      
      setState(prev => ({
        ...prev,
        tracks: [...prev.tracks, newTrack],
        selectedTrack: newTrack.id,
      }))

      addAction({
        type: 'PASTE_SELECTION',
        payload: { track: newTrack },
      })
    }
  }, [state.clipboard, addAction])

  // History operations
  const undoAction = useCallback(() => {
    if (state.historyIndex < 0) return

    const action = state.history[state.historyIndex]
    
    // Implement undo logic based on action type
    switch (action.type) {
      case 'ADD_TRACK':
        setState(prev => ({
          ...prev,
          tracks: prev.tracks.filter(t => t.id !== action.payload.track.id),
          historyIndex: prev.historyIndex - 1,
        }))
        break
      
      case 'REMOVE_TRACK':
        // Would need to restore the track from payload
        setState(prev => ({
          ...prev,
          historyIndex: prev.historyIndex - 1,
        }))
        break
      
      case 'UPDATE_TRACK':
        // Would need to reverse the update
        setState(prev => ({
          ...prev,
          historyIndex: prev.historyIndex - 1,
        }))
        break
      
      default:
        setState(prev => ({
          ...prev,
          historyIndex: prev.historyIndex - 1,
        }))
    }
  }, [state.history, state.historyIndex])

  const redoAction = useCallback(() => {
    if (state.historyIndex >= state.history.length - 1) return

    const nextIndex = state.historyIndex + 1
    const action = state.history[nextIndex]
    
    // Implement redo logic based on action type
    switch (action.type) {
      case 'ADD_TRACK':
        setState(prev => ({
          ...prev,
          tracks: [...prev.tracks, action.payload.track],
          historyIndex: nextIndex,
        }))
        break
      
      default:
        setState(prev => ({
          ...prev,
          historyIndex: nextIndex,
        }))
    }
  }, [state.history, state.historyIndex])

  // Effect operations
  const addEffect = useCallback((trackId: string, effectType: string) => {
    const newEffect: Effect = {
      id: uuidv4(),
      type: effectType,
      name: effectType.charAt(0).toUpperCase() + effectType.slice(1),
      enabled: true,
      parameters: {},
    }

    updateTrack(trackId, {
      effects: [...(state.tracks.find(t => t.id === trackId)?.effects || []), newEffect],
    })
  }, [state.tracks, updateTrack])

  const removeEffect = useCallback((trackId: string, effectId: string) => {
    const track = state.tracks.find(t => t.id === trackId)
    if (!track) return

    updateTrack(trackId, {
      effects: track.effects.filter(e => e.id !== effectId),
    })
  }, [state.tracks, updateTrack])

  const updateEffect = useCallback((trackId: string, effectId: string, updates: Partial<Effect>) => {
    const track = state.tracks.find(t => t.id === trackId)
    if (!track) return

    updateTrack(trackId, {
      effects: track.effects.map(effect =>
        effect.id === effectId ? { ...effect, ...updates } : effect
      ),
    })
  }, [state.tracks, updateTrack])

  return {
    // State
    tracks: state.tracks,
    selectedTrack: state.selectedTrack,
    canUndo: state.historyIndex >= 0,
    canRedo: state.historyIndex < state.history.length - 1,
    clipboard: state.clipboard,

    // Track operations
    addTrack,
    removeTrack,
    updateTrack,
    selectTrack,

    // Edit operations
    cutSelection,
    copySelection,
    pasteSelection,

    // History operations
    undoAction,
    redoAction,

    // Effect operations
    addEffect,
    removeEffect,
    updateEffect,
  }
}