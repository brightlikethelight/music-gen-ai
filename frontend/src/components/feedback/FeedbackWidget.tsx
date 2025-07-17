'use client'

import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ChatBubbleLeftRightIcon,
  XMarkIcon,
  StarIcon,
  PaperAirplaneIcon,
  CameraIcon,
  MicrophoneIcon,
  StopIcon,
} from '@heroicons/react/24/outline'
import { StarIcon as StarSolidIcon } from '@heroicons/react/24/solid'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'

interface FeedbackData {
  type: 'bug' | 'feature' | 'general' | 'ux'
  rating: number
  message: string
  page: string
  userAgent: string
  screenshot?: string
  audioRecording?: Blob
  metadata: {
    timestamp: number
    sessionId: string
    userId?: string
  }
}

export function FeedbackWidget() {
  const [isOpen, setIsOpen] = useState(false)
  const [step, setStep] = useState<'type' | 'rating' | 'details' | 'success'>('type')
  const [feedbackType, setFeedbackType] = useState<FeedbackData['type']>('general')
  const [rating, setRating] = useState(0)
  const [message, setMessage] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const feedbackTypes = [
    { value: 'bug', label: '🐛 Report a Bug', description: 'Something isn\'t working as expected' },
    { value: 'feature', label: '💡 Request Feature', description: 'Suggest a new feature or improvement' },
    { value: 'ux', label: '🎨 UX Feedback', description: 'Share thoughts on user experience' },
    { value: 'general', label: '💬 General Feedback', description: 'Any other thoughts or comments' },
  ]

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        setAudioBlob(audioBlob)
        stream.getTracks().forEach(track => track.stop())
      }
      
      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('Error starting recording:', error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const takeScreenshot = async () => {
    try {
      // In a real implementation, you'd use a library like html2canvas
      // For now, we'll simulate the screenshot functionality
      console.log('Screenshot functionality would be implemented here')
    } catch (error) {
      console.error('Error taking screenshot:', error)
    }
  }

  const submitFeedback = async () => {
    setIsSubmitting(true)
    
    const feedbackData: FeedbackData = {
      type: feedbackType,
      rating,
      message,
      page: window.location.pathname,
      userAgent: navigator.userAgent,
      audioRecording: audioBlob || undefined,
      metadata: {
        timestamp: Date.now(),
        sessionId: 'session-' + Date.now(), // In real app, use proper session management
        userId: undefined, // Would come from auth context
      }
    }

    try {
      // Send feedback to your backend
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData),
      })

      if (response.ok) {
        setStep('success')
        setTimeout(() => {
          resetForm()
          setIsOpen(false)
        }, 2000)
      }
    } catch (error) {
      console.error('Error submitting feedback:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const resetForm = () => {
    setStep('type')
    setFeedbackType('general')
    setRating(0)
    setMessage('')
    setAudioBlob(null)
  }

  const renderStepContent = () => {
    switch (step) {
      case 'type':
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">
              What type of feedback do you have?
            </h3>
            <div className="space-y-2">
              {feedbackTypes.map((type) => (
                <button
                  key={type.value}
                  onClick={() => {
                    setFeedbackType(type.value as FeedbackData['type'])
                    setStep('rating')
                  }}
                  className="w-full text-left p-4 rounded-lg border border-gray-600 hover:border-purple-500 hover:bg-gray-700 transition-colors"
                >
                  <div className="font-medium text-white">{type.label}</div>
                  <div className="text-sm text-gray-400">{type.description}</div>
                </button>
              ))}
            </div>
          </div>
        )

      case 'rating':
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">
              How would you rate your experience?
            </h3>
            <div className="flex justify-center space-x-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => setRating(star)}
                  className="p-1 hover:scale-110 transition-transform"
                >
                  {star <= rating ? (
                    <StarSolidIcon className="h-8 w-8 text-yellow-400" />
                  ) : (
                    <StarIcon className="h-8 w-8 text-gray-400 hover:text-yellow-400" />
                  )}
                </button>
              ))}
            </div>
            <div className="flex justify-between">
              <Button
                variant="ghost"
                onClick={() => setStep('type')}
                className="text-gray-400"
              >
                Back
              </Button>
              <Button
                onClick={() => setStep('details')}
                disabled={rating === 0}
                className="bg-purple-600 hover:bg-purple-700"
              >
                Continue
              </Button>
            </div>
          </div>
        )

      case 'details':
        return (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">
              Tell us more about your feedback
            </h3>
            
            <Textarea
              placeholder="Describe your feedback in detail..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              className="min-h-[120px] bg-gray-700 border-gray-600 text-white placeholder-gray-400"
            />

            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={takeScreenshot}
                className="flex items-center space-x-2 border-gray-600 text-gray-300 hover:bg-gray-700"
              >
                <CameraIcon className="h-4 w-4" />
                <span>Screenshot</span>
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={isRecording ? stopRecording : startRecording}
                className={`flex items-center space-x-2 border-gray-600 text-gray-300 hover:bg-gray-700 ${
                  isRecording ? 'bg-red-600 hover:bg-red-700 text-white' : ''
                }`}
              >
                {isRecording ? (
                  <StopIcon className="h-4 w-4" />
                ) : (
                  <MicrophoneIcon className="h-4 w-4" />
                )}
                <span>{isRecording ? 'Stop Recording' : 'Voice Note'}</span>
              </Button>
            </div>

            {audioBlob && (
              <div className="p-3 bg-gray-700 rounded-lg">
                <div className="text-sm text-green-400">✓ Voice recording attached</div>
              </div>
            )}

            <div className="flex justify-between">
              <Button
                variant="ghost"
                onClick={() => setStep('rating')}
                className="text-gray-400"
              >
                Back
              </Button>
              <Button
                onClick={submitFeedback}
                disabled={!message.trim() || isSubmitting}
                className="bg-purple-600 hover:bg-purple-700 flex items-center space-x-2"
              >
                {isSubmitting ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <PaperAirplaneIcon className="h-4 w-4" />
                )}
                <span>{isSubmitting ? 'Sending...' : 'Send Feedback'}</span>
              </Button>
            </div>
          </div>
        )

      case 'success':
        return (
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto">
              <div className="text-2xl">✓</div>
            </div>
            <h3 className="text-lg font-semibold text-white">
              Thank you for your feedback!
            </h3>
            <p className="text-gray-400">
              Your feedback helps us improve the MusicGen AI experience for everyone.
            </p>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <>
      {/* Feedback Button */}
      <motion.button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 bg-purple-600 hover:bg-purple-700 text-white p-4 rounded-full shadow-lg transition-colors"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <ChatBubbleLeftRightIcon className="h-6 w-6" />
      </motion.button>

      {/* Feedback Modal */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 bg-black bg-opacity-50 z-50"
            />

            {/* Modal */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-4"
            >
              <div className="bg-gray-800 rounded-2xl border border-gray-700 w-full max-w-md max-h-[90vh] overflow-y-auto">
                <div className="flex items-center justify-between p-6 border-b border-gray-700">
                  <h2 className="text-xl font-semibold text-white">Feedback</h2>
                  <button
                    onClick={() => setIsOpen(false)}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <XMarkIcon className="h-6 w-6" />
                  </button>
                </div>
                
                <div className="p-6">
                  {renderStepContent()}
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  )
}