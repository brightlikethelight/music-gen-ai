'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MusicalNoteIcon, 
  PlayIcon, 
  PauseIcon, 
  StopIcon,
  ArrowDownTrayIcon,
  ShareIcon,
  Cog6ToothIcon,
  SparklesIcon
} from '@heroicons/react/24/outline'
import { PlayIcon as PlaySolidIcon } from '@heroicons/react/24/solid'

import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Textarea'
import { Slider } from '@/components/ui/Slider'
import { Select } from '@/components/ui/Select'
import { WaveformPlayer } from '@/components/audio/WaveformPlayer'
import { GenerationProgress } from '@/components/generation/GenerationProgress'
import { ParameterControls } from '@/components/generation/ParameterControls'
import { GenerationHistory } from '@/components/generation/GenerationHistory'
import { useGeneration } from '@/hooks/useGeneration'
import { useAudio } from '@/hooks/useAudio'
import { useWebSocket } from '@/hooks/useWebSocket'

const genreOptions = [
  { value: 'pop', label: 'Pop' },
  { value: 'rock', label: 'Rock' },
  { value: 'jazz', label: 'Jazz' },
  { value: 'electronic', label: 'Electronic' },
  { value: 'classical', label: 'Classical' },
  { value: 'hip-hop', label: 'Hip-Hop' },
  { value: 'country', label: 'Country' },
  { value: 'r&b', label: 'R&B' },
  { value: 'folk', label: 'Folk' },
  { value: 'ambient', label: 'Ambient' },
]

const moodOptions = [
  { value: 'happy', label: 'Happy' },
  { value: 'sad', label: 'Sad' },
  { value: 'energetic', label: 'Energetic' },
  { value: 'calm', label: 'Calm' },
  { value: 'aggressive', label: 'Aggressive' },
  { value: 'romantic', label: 'Romantic' },
  { value: 'mysterious', label: 'Mysterious' },
  { value: 'uplifting', label: 'Uplifting' },
]

export function GenerationStudio() {
  const [prompt, setPrompt] = useState('')
  const [genre, setGenre] = useState('pop')
  const [mood, setMood] = useState('happy')
  const [duration, setDuration] = useState(30)
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  const {
    isGenerating,
    generationProgress,
    currentTrack,
    generationHistory,
    generateMusic,
    stopGeneration,
    saveGeneration,
  } = useGeneration()
  
  const {
    isPlaying,
    currentTime,
    duration: trackDuration,
    togglePlayback,
    seek,
    stop,
  } = useAudio()
  
  const { isConnected } = useWebSocket()

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    const parameters = {
      prompt: prompt.trim(),
      genre,
      mood,
      duration,
      // Advanced parameters would be included here
    }

    await generateMusic(parameters)
  }

  const handlePromptChange = (value: string) => {
    setPrompt(value)
  }

  const promptPlaceholders = [
    "Upbeat jazz quartet with saxophone solo",
    "Ambient electronic soundscape for relaxation",
    "Epic orchestral theme with soaring strings",
    "Acoustic folk song with gentle guitar picking",
    "Energetic rock anthem with powerful drums",
    "Smooth R&B track with soulful vocals",
    "Classical piano piece in minor key",
    "Future bass drop with synthesizer leads",
  ]

  const [placeholderIndex, setPlaceholderIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setPlaceholderIndex((prev) => (prev + 1) % promptPlaceholders.length)
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Main Generation Panel */}
        <div className="lg:col-span-8">
          <Card className="p-6">
            <div className="mb-6 flex items-center justify-between">
              <h1 className="text-2xl font-bold text-gray-900">
                Music Generation Studio
              </h1>
              <div className="flex items-center space-x-2">
                <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm text-gray-500">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>

            {/* Prompt Input */}
            <div className="mb-6">
              <label htmlFor="prompt" className="mb-2 block text-sm font-medium text-gray-700">
                Describe your music
              </label>
              <Textarea
                id="prompt"
                value={prompt}
                onChange={(e) => handlePromptChange(e.target.value)}
                placeholder={promptPlaceholders[placeholderIndex]}
                rows={3}
                className="w-full"
                disabled={isGenerating}
              />
              <p className="mt-1 text-xs text-gray-500">
                Be specific about style, instruments, mood, and any other details you want in your music.
              </p>
            </div>

            {/* Quick Settings */}
            <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div>
                <label htmlFor="genre" className="mb-2 block text-sm font-medium text-gray-700">
                  Genre
                </label>
                <Select
                  id="genre"
                  value={genre}
                  onChange={setGenre}
                  options={genreOptions}
                  disabled={isGenerating}
                />
              </div>
              
              <div>
                <label htmlFor="mood" className="mb-2 block text-sm font-medium text-gray-700">
                  Mood
                </label>
                <Select
                  id="mood"
                  value={mood}
                  onChange={setMood}
                  options={moodOptions}
                  disabled={isGenerating}
                />
              </div>
              
              <div>
                <label htmlFor="duration" className="mb-2 block text-sm font-medium text-gray-700">
                  Duration: {duration}s
                </label>
                <Slider
                  id="duration"
                  value={[duration]}
                  onValueChange={(value) => setDuration(value[0])}
                  min={10}
                  max={300}
                  step={10}
                  disabled={isGenerating}
                  className="mt-2"
                />
              </div>
            </div>

            {/* Advanced Parameters Toggle */}
            <div className="mb-6">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center space-x-2 text-sm text-blue-600 hover:text-blue-700"
                disabled={isGenerating}
              >
                <Cog6ToothIcon className="h-4 w-4" />
                <span>Advanced Parameters</span>
              </button>
            </div>

            {/* Advanced Parameters */}
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mb-6"
                >
                  <ParameterControls disabled={isGenerating} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Generation Button */}
            <div className="mb-6">
              <Button
                onClick={handleGenerate}
                disabled={isGenerating || !prompt.trim()}
                size="lg"
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700"
              >
                {isGenerating ? (
                  <>
                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Generating...
                  </>
                ) : (
                  <>
                    <SparklesIcon className="mr-2 h-5 w-5" />
                    Generate Music
                  </>
                )}
              </Button>
            </div>

            {/* Generation Progress */}
            <AnimatePresence>
              {isGenerating && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="mb-6"
                >
                  <GenerationProgress
                    progress={generationProgress}
                    onCancel={stopGeneration}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Audio Player */}
            <AnimatePresence>
              {currentTrack && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="mb-6"
                >
                  <Card className="p-4">
                    <div className="mb-4 flex items-center justify-between">
                      <div>
                        <h3 className="font-medium text-gray-900">Generated Track</h3>
                        <p className="text-sm text-gray-500">{currentTrack.prompt}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => saveGeneration(currentTrack.id)}
                        >
                          <ArrowDownTrayIcon className="mr-1 h-4 w-4" />
                          Save
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                        >
                          <ShareIcon className="mr-1 h-4 w-4" />
                          Share
                        </Button>
                      </div>
                    </div>

                    <WaveformPlayer
                      src={currentTrack.audioUrl}
                      waveformData={currentTrack.waveformData}
                      isPlaying={isPlaying}
                      currentTime={currentTime}
                      duration={trackDuration}
                      onPlayPause={togglePlayback}
                      onSeek={seek}
                      onStop={stop}
                    />
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-4">
          <div className="space-y-6">
            {/* Generation History */}
            <Card className="p-4">
              <h2 className="mb-4 text-lg font-medium text-gray-900">Recent Generations</h2>
              <GenerationHistory
                generations={generationHistory}
                onSelect={(track) => {
                  // Load selected track
                }}
              />
            </Card>

            {/* Quick Tips */}
            <Card className="p-4">
              <h2 className="mb-4 text-lg font-medium text-gray-900">Pro Tips</h2>
              <div className="space-y-3">
                <div className="flex items-start space-x-2">
                  <div className="mt-1 h-2 w-2 rounded-full bg-blue-500" />
                  <p className="text-sm text-gray-600">
                    Be specific about instruments you want to hear
                  </p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="mt-1 h-2 w-2 rounded-full bg-blue-500" />
                  <p className="text-sm text-gray-600">
                    Mention the energy level or tempo you prefer
                  </p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="mt-1 h-2 w-2 rounded-full bg-blue-500" />
                  <p className="text-sm text-gray-600">
                    Include musical structure (verse, chorus, bridge)
                  </p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="mt-1 h-2 w-2 rounded-full bg-blue-500" />
                  <p className="text-sm text-gray-600">
                    Try different combinations of genre and mood
                  </p>
                </div>
              </div>
            </Card>

            {/* Community Showcase */}
            <Card className="p-4">
              <h2 className="mb-4 text-lg font-medium text-gray-900">Community Favorites</h2>
              <div className="space-y-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div key={i} className="flex items-center space-x-3">
                    <div className="h-10 w-10 rounded-full bg-gradient-to-br from-blue-400 to-purple-500" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        Amazing Track #{i + 1}
                      </p>
                      <p className="text-xs text-gray-500">by User{i + 1}</p>
                    </div>
                    <button className="text-gray-400 hover:text-gray-600">
                      <PlaySolidIcon className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}