'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { PlayIcon, SparklesIcon, MusicalNoteIcon } from '@heroicons/react/24/solid'
import { Button } from '@/components/ui/Button'
import { WaveformVisualization } from '@/components/audio/WaveformVisualization'
import { useAudio } from '@/hooks/useAudio'

const demoTracks = [
  {
    id: '1',
    title: 'Ethereal Ambient',
    description: 'Generated with: "Dreamy ambient soundscape with gentle pads"',
    duration: 30,
    waveformData: Array.from({ length: 100 }, () => Math.random() * 0.8 + 0.2),
  },
  {
    id: '2', 
    title: 'Epic Orchestral',
    description: 'Generated with: "Cinematic orchestral theme with soaring strings"',
    duration: 45,
    waveformData: Array.from({ length: 100 }, () => Math.random() * 1.0 + 0.1),
  },
  {
    id: '3',
    title: 'Future Bass Drop',
    description: 'Generated with: "Energetic future bass with powerful drop"',
    duration: 60,
    waveformData: Array.from({ length: 100 }, () => Math.random() * 0.9 + 0.3),
  },
]

export function HeroSection() {
  const [currentTrack, setCurrentTrack] = useState(0)
  const [isGenerating, setIsGenerating] = useState(false)
  const { isPlaying, togglePlayback } = useAudio()

  useEffect(() => {
    const interval = setInterval(() => {
      if (!isPlaying) {
        setCurrentTrack((prev) => (prev + 1) % demoTracks.length)
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [isPlaying])

  const handleDemoGeneration = () => {
    setIsGenerating(true)
    setTimeout(() => {
      setIsGenerating(false)
    }, 3000)
  }

  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-music-pattern opacity-5" />
      
      {/* Animated Background Elements */}
      <div className="absolute inset-0">
        {Array.from({ length: 20 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute h-2 w-2 rounded-full bg-blue-400"
            animate={{
              x: [0, Math.random() * 100 - 50],
              y: [0, Math.random() * 100 - 50],
              opacity: [0, 1, 0],
            }}
            transition={{
              duration: Math.random() * 10 + 10,
              repeat: Infinity,
              ease: 'linear',
            }}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
          />
        ))}
      </div>

      <div className="relative px-6 py-24 sm:px-8 sm:py-32 lg:px-12">
        <div className="mx-auto max-w-7xl">
          <div className="grid grid-cols-1 gap-12 lg:grid-cols-2 lg:gap-20">
            {/* Left Column - Content */}
            <div className="flex flex-col justify-center">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
              >
                <div className="mb-6 flex items-center space-x-2">
                  <SparklesIcon className="h-6 w-6 text-yellow-400" />
                  <span className="text-sm font-medium text-blue-200">
                    Professional AI Music Generation
                  </span>
                </div>

                <h1 className="mb-6 text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
                  Create Studio-Quality
                  <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                    {' '}Music with AI
                  </span>
                </h1>

                <p className="mb-8 text-lg text-blue-100 sm:text-xl lg:text-2xl">
                  Generate professional compositions, collaborate in real-time, and bring your musical ideas to life with our advanced AI platform.
                </p>

                <div className="mb-8 flex flex-col space-y-4 sm:flex-row sm:space-x-4 sm:space-y-0">
                  <Button
                    size="lg"
                    className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:from-blue-600 hover:to-cyan-600"
                    asChild
                  >
                    <Link href="/studio/generate">
                      <MusicalNoteIcon className="mr-2 h-5 w-5" />
                      Start Creating
                    </Link>
                  </Button>

                  <Button
                    size="lg"
                    variant="outline"
                    className="border-blue-300 text-blue-100 hover:bg-blue-800/50"
                    onClick={handleDemoGeneration}
                    disabled={isGenerating}
                  >
                    <PlayIcon className="mr-2 h-5 w-5" />
                    {isGenerating ? 'Generating...' : 'Watch Demo'}
                  </Button>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">50K+</div>
                    <div className="text-sm text-blue-200">Tracks Created</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">10K+</div>
                    <div className="text-sm text-blue-200">Active Users</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white">99.9%</div>
                    <div className="text-sm text-blue-200">Uptime</div>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Right Column - Interactive Demo */}
            <div className="flex items-center justify-center">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="w-full max-w-md"
              >
                <div className="rounded-2xl bg-white/10 p-6 backdrop-blur-lg">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white">
                      Live Demo
                    </h3>
                    <div className="flex space-x-1">
                      {demoTracks.map((_, index) => (
                        <button
                          key={index}
                          className={`h-2 w-8 rounded-full transition-colors ${
                            index === currentTrack ? 'bg-blue-400' : 'bg-white/30'
                          }`}
                          onClick={() => setCurrentTrack(index)}
                        />
                      ))}
                    </div>
                  </div>

                  <div className="mb-4">
                    <h4 className="mb-2 font-medium text-white">
                      {demoTracks[currentTrack].title}
                    </h4>
                    <p className="text-sm text-blue-200">
                      {demoTracks[currentTrack].description}
                    </p>
                  </div>

                  {/* Waveform Visualization */}
                  <div className="mb-4 h-24 rounded-lg bg-black/20 p-2">
                    <WaveformVisualization
                      data={demoTracks[currentTrack].waveformData}
                      isPlaying={isPlaying}
                      className="h-full w-full"
                    />
                  </div>

                  {/* Playback Controls */}
                  <div className="flex items-center space-x-4">
                    <button
                      onClick={togglePlayback}
                      className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-500 text-white transition-colors hover:bg-blue-600"
                    >
                      <PlayIcon className="h-6 w-6" />
                    </button>

                    <div className="flex-1">
                      <div className="mb-1 flex items-center justify-between text-xs text-blue-200">
                        <span>0:00</span>
                        <span>{Math.floor(demoTracks[currentTrack].duration / 60)}:{(demoTracks[currentTrack].duration % 60).toString().padStart(2, '0')}</span>
                      </div>
                      <div className="h-1 rounded-full bg-white/20">
                        <motion.div
                          className="h-1 rounded-full bg-blue-400"
                          initial={{ width: '0%' }}
                          animate={{ width: isPlaying ? '100%' : '0%' }}
                          transition={{ duration: demoTracks[currentTrack].duration, ease: 'linear' }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Generation Status */}
                  {isGenerating && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 rounded-lg bg-green-500/20 p-3"
                    >
                      <div className="flex items-center space-x-2">
                        <div className="h-2 w-2 animate-pulse rounded-full bg-green-400" />
                        <span className="text-sm text-green-300">
                          Generating your music...
                        </span>
                      </div>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}