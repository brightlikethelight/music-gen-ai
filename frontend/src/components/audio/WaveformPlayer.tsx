'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  ArrowDownTrayIcon,
  ShareIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { Slider } from '@/components/ui/Slider'
import { formatTime } from '@/utils/time'

interface WaveformPlayerProps {
  src: string
  waveformData?: number[]
  isPlaying?: boolean
  currentTime?: number
  duration?: number
  className?: string
  onPlayPause?: () => void
  onSeek?: (time: number) => void
  onStop?: () => void
  onVolumeChange?: (volume: number) => void
  showControls?: boolean
  showWaveform?: boolean
  height?: number
  barWidth?: number
  barGap?: number
  progressColor?: string
  waveformColor?: string
  backgroundColor?: string
}

export function WaveformPlayer({
  src,
  waveformData = [],
  isPlaying = false,
  currentTime = 0,
  duration = 0,
  className = '',
  onPlayPause,
  onSeek,
  onStop,
  onVolumeChange,
  showControls = true,
  showWaveform = true,
  height = 80,
  barWidth = 2,
  barGap = 1,
  progressColor = '#3b82f6',
  waveformColor = '#e5e7eb',
  backgroundColor = '#f8fafc',
}: WaveformPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [volume, setVolume] = useState(70)
  const [isMuted, setIsMuted] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })

  // Resize observer for responsive canvas
  useEffect(() => {
    if (!containerRef.current) return

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect
        setCanvasSize({ width, height })
      }
    })

    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
    }
  }, [height])

  // Generate default waveform data if none provided
  const normalizedWaveformData = waveformData.length > 0 
    ? waveformData 
    : Array.from({ length: 100 }, () => Math.random() * 0.8 + 0.1)

  // Draw waveform on canvas
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const { width, height } = canvasSize
    if (width === 0 || height === 0) return

    // Set canvas size
    canvas.width = width * window.devicePixelRatio
    canvas.height = height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    // Clear canvas
    ctx.fillStyle = backgroundColor
    ctx.fillRect(0, 0, width, height)

    // Calculate bar dimensions
    const totalBars = Math.floor(width / (barWidth + barGap))
    const dataPointsPerBar = Math.max(1, Math.floor(normalizedWaveformData.length / totalBars))

    // Calculate progress position
    const progressPosition = duration > 0 ? (currentTime / duration) * width : 0

    // Draw waveform bars
    for (let i = 0; i < totalBars; i++) {
      const x = i * (barWidth + barGap)
      
      // Get average amplitude for this bar
      const startIdx = i * dataPointsPerBar
      const endIdx = Math.min(startIdx + dataPointsPerBar, normalizedWaveformData.length)
      const amplitude = normalizedWaveformData
        .slice(startIdx, endIdx)
        .reduce((sum, val) => sum + val, 0) / (endIdx - startIdx)

      const barHeight = Math.max(2, amplitude * height * 0.8)
      const barY = (height - barHeight) / 2

      // Determine bar color based on progress
      const isPlayed = x < progressPosition
      ctx.fillStyle = isPlayed ? progressColor : waveformColor

      // Draw bar with rounded corners
      ctx.beginPath()
      ctx.roundRect(x, barY, barWidth, barHeight, barWidth / 2)
      ctx.fill()
    }

    // Draw progress indicator
    if (isPlaying && progressPosition > 0) {
      ctx.fillStyle = progressColor
      ctx.fillRect(progressPosition - 1, 0, 2, height)
    }
  }, [
    canvasSize,
    normalizedWaveformData,
    currentTime,
    duration,
    isPlaying,
    barWidth,
    barGap,
    progressColor,
    waveformColor,
    backgroundColor,
  ])

  // Redraw waveform when dependencies change
  useEffect(() => {
    drawWaveform()
  }, [drawWaveform])

  // Handle canvas click for seeking
  const handleCanvasClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek || duration === 0) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const seekTime = (x / rect.width) * duration
    
    onSeek(Math.max(0, Math.min(seekTime, duration)))
  }, [onSeek, duration])

  // Handle volume change
  const handleVolumeChange = useCallback((newVolume: number[]) => {
    const volumeValue = newVolume[0]
    setVolume(volumeValue)
    setIsMuted(volumeValue === 0)
    onVolumeChange?.(volumeValue / 100)
  }, [onVolumeChange])

  // Toggle mute
  const handleMuteToggle = useCallback(() => {
    setIsMuted(!isMuted)
    onVolumeChange?.(isMuted ? volume / 100 : 0)
  }, [isMuted, volume, onVolumeChange])

  return (
    <div className={`rounded-lg bg-white p-4 shadow-sm ${className}`}>
      {/* Waveform Display */}
      {showWaveform && (
        <div
          ref={containerRef}
          className="relative mb-4 cursor-pointer overflow-hidden rounded-md"
          style={{ height: `${height}px` }}
        >
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            className="absolute inset-0 h-full w-full"
            style={{ width: '100%', height: '100%' }}
          />
          
          {/* Loading overlay */}
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/20">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      {showControls && (
        <div className="space-y-3">
          {/* Main Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Button
                size="sm"
                onClick={onPlayPause}
                className="h-10 w-10 rounded-full p-0"
                disabled={!src}
              >
                {isPlaying ? (
                  <PauseIcon className="h-5 w-5" />
                ) : (
                  <PlayIcon className="h-5 w-5" />
                )}
              </Button>

              <Button
                size="sm"
                variant="outline"
                onClick={onStop}
                className="h-8 w-8 rounded-full p-0"
                disabled={!src}
              >
                <StopIcon className="h-4 w-4" />
              </Button>

              {/* Time Display */}
              <div className="text-sm text-gray-500">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Volume Control */}
              <button
                onClick={handleMuteToggle}
                className="text-gray-400 hover:text-gray-600"
              >
                {isMuted || volume === 0 ? (
                  <SpeakerXMarkIcon className="h-5 w-5" />
                ) : (
                  <SpeakerWaveIcon className="h-5 w-5" />
                )}
              </button>

              <div className="w-20">
                <Slider
                  value={[isMuted ? 0 : volume]}
                  onValueChange={handleVolumeChange}
                  max={100}
                  step={1}
                  className="w-full"
                />
              </div>

              {/* Additional Controls */}
              <Button size="sm" variant="outline">
                <ArrowDownTrayIcon className="h-4 w-4" />
              </Button>

              <Button size="sm" variant="outline">
                <ShareIcon className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-1">
            <Slider
              value={[currentTime]}
              onValueChange={(value) => onSeek?.(value[0])}
              max={duration}
              step={0.1}
              className="w-full"
              disabled={!src || duration === 0}
            />
          </div>
        </div>
      )}
    </div>
  )
}