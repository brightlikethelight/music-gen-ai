'use client'

import { useEffect, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'

interface WaveformVisualizationProps {
  data: number[]
  isPlaying?: boolean
  currentTime?: number
  duration?: number
  className?: string
  height?: number
  color?: string
  backgroundColor?: string
  animateOnPlay?: boolean
  showProgress?: boolean
  barCount?: number
}

export function WaveformVisualization({
  data,
  isPlaying = false,
  currentTime = 0,
  duration = 1,
  className = '',
  height = 60,
  color = '#3b82f6',
  backgroundColor = 'transparent',
  animateOnPlay = true,
  showProgress = true,
  barCount = 50,
}: WaveformVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Normalize data to fit the specified bar count
  const normalizedData = useCallback(() => {
    if (data.length === 0) {
      return Array.from({ length: barCount }, () => Math.random() * 0.8 + 0.1)
    }

    if (data.length === barCount) {
      return data
    }

    const result: number[] = []
    const chunkSize = data.length / barCount

    for (let i = 0; i < barCount; i++) {
      const start = Math.floor(i * chunkSize)
      const end = Math.floor((i + 1) * chunkSize)
      const chunk = data.slice(start, end)
      const average = chunk.reduce((sum, val) => sum + val, 0) / chunk.length
      result.push(average)
    }

    return result
  }, [data, barCount])

  const drawWaveform = useCallback((timestamp: number = 0) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const width = rect.width
    const canvasHeight = height

    // Set canvas size accounting for device pixel ratio
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = canvasHeight * dpr
    ctx.scale(dpr, dpr)

    // Clear canvas
    ctx.fillStyle = backgroundColor
    ctx.fillRect(0, 0, width, canvasHeight)

    const normalizedWaveform = normalizedData()
    const barWidth = width / normalizedWaveform.length
    const progressPosition = showProgress && duration > 0 ? (currentTime / duration) * width : 0

    // Draw waveform bars
    normalizedWaveform.forEach((amplitude, index) => {
      const x = index * barWidth
      const barHeight = Math.max(2, amplitude * canvasHeight * 0.8)
      const y = (canvasHeight - barHeight) / 2

      // Determine bar color based on progress and animation
      let barColor = color
      const isPlayed = showProgress && x < progressPosition

      if (isPlayed) {
        barColor = color
      } else {
        // Dim unplayed bars
        barColor = `${color}40` // Add transparency
      }

      // Add animation effect when playing
      if (animateOnPlay && isPlaying) {
        const animationOffset = Math.sin(timestamp * 0.005 + index * 0.2) * 0.1
        const animatedHeight = barHeight * (1 + animationOffset)
        const animatedY = (canvasHeight - animatedHeight) / 2

        ctx.fillStyle = barColor
        ctx.fillRect(x, animatedY, barWidth - 1, animatedHeight)
      } else {
        ctx.fillStyle = barColor
        ctx.fillRect(x, y, barWidth - 1, barHeight)
      }
    })

    // Draw progress indicator line
    if (showProgress && progressPosition > 0) {
      ctx.fillStyle = color
      ctx.fillRect(progressPosition - 1, 0, 2, canvasHeight)
    }

    // Continue animation if playing
    if (animateOnPlay && isPlaying) {
      animationRef.current = requestAnimationFrame(drawWaveform)
    }
  }, [
    normalizedData,
    height,
    color,
    backgroundColor,
    currentTime,
    duration,
    showProgress,
    animateOnPlay,
    isPlaying,
  ])

  // Start/stop animation based on playing state
  useEffect(() => {
    if (animateOnPlay && isPlaying) {
      animationRef.current = requestAnimationFrame(drawWaveform)
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      drawWaveform()
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying, animateOnPlay, drawWaveform])

  // Redraw when data or other props change
  useEffect(() => {
    if (!animateOnPlay || !isPlaying) {
      drawWaveform()
    }
  }, [data, currentTime, duration, color, backgroundColor, drawWaveform, animateOnPlay, isPlaying])

  return (
    <div className={`relative overflow-hidden rounded ${className}`} style={{ height: `${height}px` }}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full"
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* Overlay effects */}
      {isPlaying && animateOnPlay && (
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
          animate={{
            x: [-100, '100%'],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: 'linear',
          }}
          style={{
            width: '20%',
          }}
        />
      )}
    </div>
  )
}