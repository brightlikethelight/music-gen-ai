'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  BackwardIcon,
  ForwardIcon,
  ScissorsIcon,
  DocumentDuplicateIcon,
  AdjustmentsHorizontalIcon,
  SpeakerWaveIcon,
  Cog6ToothIcon,
  ArrowDownTrayIcon,
  FolderOpenIcon,
  PlusIcon,
  MinusIcon,
} from '@heroicons/react/24/outline'

import { Button } from '@/components/ui/Button'
import { Slider } from '@/components/ui/Slider'
import { Select } from '@/components/ui/Select'
import { Timeline } from '@/components/editor/Timeline'
import { TrackPanel } from '@/components/editor/TrackPanel'
import { EffectsPanel } from '@/components/editor/EffectsPanel'
import { MixerPanel } from '@/components/editor/MixerPanel'
import { ToolsPanel } from '@/components/editor/ToolsPanel'
import { ExportDialog } from '@/components/editor/ExportDialog'
import { useAudioEditor } from '@/hooks/useAudioEditor'
import { formatTime } from '@/utils/time'

interface Track {
  id: string
  name: string
  type: 'audio' | 'midi'
  audioUrl?: string
  muted: boolean
  solo: boolean
  volume: number
  pan: number
  effects: any[]
  waveformData: number[]
  color: string
}

export function AudioEditor() {
  const [selectedTool, setSelectedTool] = useState('select')
  const [zoomLevel, setZoomLevel] = useState(1)
  const [showExportDialog, setShowExportDialog] = useState(false)
  const [activePanel, setActivePanel] = useState<'tracks' | 'effects' | 'mixer' | 'tools'>('tracks')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(180) // 3 minutes
  const [tempo, setTempo] = useState(120)
  const [timeSignature, setTimeSignature] = useState('4/4')

  const {
    tracks,
    selectedTrack,
    addTrack,
    removeTrack,
    updateTrack,
    cutSelection,
    copySelection,
    pasteSelection,
    undoAction,
    redoAction,
    canUndo,
    canRedo,
  } = useAudioEditor()

  // Transport controls
  const handlePlay = () => {
    setIsPlaying(!isPlaying)
  }

  const handleStop = () => {
    setIsPlaying(false)
    setCurrentTime(0)
  }

  const handleRewind = () => {
    setCurrentTime(Math.max(0, currentTime - 10))
  }

  const handleForward = () => {
    setCurrentTime(Math.min(duration, currentTime + 10))
  }

  // Timeline controls
  const handleTimelineClick = (time: number) => {
    setCurrentTime(time)
  }

  const handleZoomIn = () => {
    setZoomLevel(Math.min(8, zoomLevel * 1.5))
  }

  const handleZoomOut = () => {
    setZoomLevel(Math.max(0.25, zoomLevel / 1.5))
  }

  // File operations
  const handleOpenProject = () => {
    // Open file dialog
    console.log('Opening project...')
  }

  const handleSaveProject = () => {
    // Save project
    console.log('Saving project...')
  }

  const handleImportAudio = () => {
    // Import audio file
    console.log('Importing audio...')
  }

  const handleExport = () => {
    setShowExportDialog(true)
  }

  const tools = [
    { id: 'select', name: 'Select', icon: '↖' },
    { id: 'cut', name: 'Cut', icon: '✂' },
    { id: 'fade', name: 'Fade', icon: '📈' },
    { id: 'pitch', name: 'Pitch', icon: '🎵' },
    { id: 'time', name: 'Time Stretch', icon: '⏱' },
  ]

  const panels = [
    { id: 'tracks', name: 'Tracks', icon: SpeakerWaveIcon },
    { id: 'effects', name: 'Effects', icon: AdjustmentsHorizontalIcon },
    { id: 'mixer', name: 'Mixer', icon: Cog6ToothIcon },
    { id: 'tools', name: 'Tools', icon: ScissorsIcon },
  ]

  return (
    <div className="flex h-screen flex-col bg-gray-900 text-white">
      {/* Top Menu Bar */}
      <div className="flex items-center justify-between border-b border-gray-700 bg-gray-800 px-4 py-2">
        <div className="flex items-center space-x-4">
          <h1 className="text-lg font-semibold">Audio Editor</h1>
          
          <div className="flex items-center space-x-2">
            <Button size="sm" variant="outline" onClick={handleOpenProject}>
              <FolderOpenIcon className="mr-1 h-4 w-4" />
              Open
            </Button>
            <Button size="sm" variant="outline" onClick={handleSaveProject}>
              Save
            </Button>
            <Button size="sm" variant="outline" onClick={handleImportAudio}>
              Import
            </Button>
            <Button size="sm" variant="outline" onClick={handleExport}>
              <ArrowDownTrayIcon className="mr-1 h-4 w-4" />
              Export
            </Button>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Tempo & Time Signature */}
          <div className="flex items-center space-x-2 text-sm">
            <span>Tempo:</span>
            <input
              type="number"
              value={tempo}
              onChange={(e) => setTempo(Number(e.target.value))}
              className="w-16 rounded bg-gray-700 px-2 py-1 text-center"
              min="60"
              max="200"
            />
            <span>BPM</span>
          </div>

          <div className="flex items-center space-x-2 text-sm">
            <span>Time:</span>
            <select
              value={timeSignature}
              onChange={(e) => setTimeSignature(e.target.value)}
              className="rounded bg-gray-700 px-2 py-1"
            >
              <option value="4/4">4/4</option>
              <option value="3/4">3/4</option>
              <option value="6/8">6/8</option>
              <option value="2/4">2/4</option>
            </select>
          </div>

          {/* Undo/Redo */}
          <div className="flex items-center space-x-1">
            <Button
              size="sm"
              variant="outline"
              onClick={undoAction}
              disabled={!canUndo}
            >
              ↶
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={redoAction}
              disabled={!canRedo}
            >
              ↷
            </Button>
          </div>
        </div>
      </div>

      {/* Transport Controls */}
      <div className="flex items-center justify-between border-b border-gray-700 bg-gray-800 px-4 py-3">
        <div className="flex items-center space-x-3">
          <Button size="sm" onClick={handleRewind}>
            <BackwardIcon className="h-4 w-4" />
          </Button>
          <Button size="sm" onClick={handlePlay}>
            {isPlaying ? (
              <PauseIcon className="h-5 w-5" />
            ) : (
              <PlayIcon className="h-5 w-5" />
            )}
          </Button>
          <Button size="sm" onClick={handleStop}>
            <StopIcon className="h-4 w-4" />
          </Button>
          <Button size="sm" onClick={handleForward}>
            <ForwardIcon className="h-4 w-4" />
          </Button>
        </div>

        {/* Time Display */}
        <div className="flex items-center space-x-4">
          <div className="text-sm font-mono">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
          
          {/* Zoom Controls */}
          <div className="flex items-center space-x-2">
            <Button size="sm" variant="outline" onClick={handleZoomOut}>
              <MinusIcon className="h-4 w-4" />
            </Button>
            <span className="text-sm">{Math.round(zoomLevel * 100)}%</span>
            <Button size="sm" variant="outline" onClick={handleZoomIn}>
              <PlusIcon className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Tools */}
        <div className="flex items-center space-x-1">
          {tools.map((tool) => (
            <button
              key={tool.id}
              onClick={() => setSelectedTool(tool.id)}
              className={`rounded px-3 py-1 text-sm transition-colors ${
                selectedTool === tool.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              title={tool.name}
            >
              {tool.icon}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Side Panel */}
        <div className="w-80 border-r border-gray-700 bg-gray-800">
          {/* Panel Tabs */}
          <div className="flex border-b border-gray-700">
            {panels.map((panel) => {
              const Icon = panel.icon
              return (
                <button
                  key={panel.id}
                  onClick={() => setActivePanel(panel.id as any)}
                  className={`flex flex-1 items-center justify-center space-x-2 py-3 text-sm transition-colors ${
                    activePanel === panel.id
                      ? 'bg-gray-700 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{panel.name}</span>
                </button>
              )
            })}
          </div>

          {/* Panel Content */}
          <div className="h-full overflow-y-auto p-4">
            <AnimatePresence mode="wait">
              {activePanel === 'tracks' && (
                <motion.div
                  key="tracks"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                >
                  <TrackPanel
                    tracks={tracks}
                    selectedTrack={selectedTrack}
                    onAddTrack={addTrack}
                    onRemoveTrack={removeTrack}
                    onUpdateTrack={updateTrack}
                  />
                </motion.div>
              )}

              {activePanel === 'effects' && (
                <motion.div
                  key="effects"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                >
                  <EffectsPanel
                    selectedTrack={selectedTrack}
                    onUpdateTrack={updateTrack}
                  />
                </motion.div>
              )}

              {activePanel === 'mixer' && (
                <motion.div
                  key="mixer"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                >
                  <MixerPanel
                    tracks={tracks}
                    onUpdateTrack={updateTrack}
                  />
                </motion.div>
              )}

              {activePanel === 'tools' && (
                <motion.div
                  key="tools"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                >
                  <ToolsPanel
                    selectedTool={selectedTool}
                    onToolChange={setSelectedTool}
                    onCut={cutSelection}
                    onCopy={copySelection}
                    onPaste={pasteSelection}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Timeline Area */}
        <div className="flex flex-1 flex-col">
          <Timeline
            tracks={tracks}
            duration={duration}
            currentTime={currentTime}
            zoomLevel={zoomLevel}
            selectedTool={selectedTool}
            isPlaying={isPlaying}
            onTimeChange={handleTimelineClick}
            onTrackUpdate={updateTrack}
          />
        </div>
      </div>

      {/* Export Dialog */}
      <ExportDialog
        isOpen={showExportDialog}
        onClose={() => setShowExportDialog(false)}
        tracks={tracks}
        duration={duration}
      />
    </div>
  )
}