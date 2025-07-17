'use client'

import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  XMarkIcon,
  FolderIcon,
  TagIcon,
  GlobeAltIcon,
  LockClosedIcon,
  MusicalNoteIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Textarea } from '@/components/ui/Textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Card, CardContent } from '@/components/ui/Card'

interface CreateProjectModalProps {
  isOpen: boolean
  onClose: () => void
  onCreate: (data: {
    name: string
    description?: string
    isPublic?: boolean
    tags?: string[]
    templateId?: string
  }) => Promise<any>
}

interface ProjectTemplate {
  id: string
  name: string
  description: string
  genre: string
  trackCount: number
  duration: string
  preview?: string
  isOfficial: boolean
}

// Mock templates - in real implementation, fetch from API
const projectTemplates: ProjectTemplate[] = [
  {
    id: 'electronic-starter',
    name: 'Electronic Starter',
    description: 'Basic electronic music template with synthesizers and drums',
    genre: 'Electronic',
    trackCount: 4,
    duration: '2:30',
    isOfficial: true,
  },
  {
    id: 'pop-ballad',
    name: 'Pop Ballad',
    description: 'Emotional pop ballad structure with piano and strings',
    genre: 'Pop',
    trackCount: 6,
    duration: '3:45',
    isOfficial: true,
  },
  {
    id: 'jazz-ensemble',
    name: 'Jazz Ensemble',
    description: 'Traditional jazz setup with piano, bass, drums, and horns',
    genre: 'Jazz',
    trackCount: 5,
    duration: '4:20',
    isOfficial: true,
  },
  {
    id: 'ambient-soundscape',
    name: 'Ambient Soundscape',
    description: 'Atmospheric ambient template with evolving textures',
    genre: 'Ambient',
    trackCount: 3,
    duration: '5:00',
    isOfficial: true,
  },
  {
    id: 'rock-band',
    name: 'Rock Band',
    description: 'Classic rock band setup with guitars, bass, and drums',
    genre: 'Rock',
    trackCount: 5,
    duration: '3:30',
    isOfficial: true,
  },
  {
    id: 'hip-hop-beat',
    name: 'Hip-Hop Beat',
    description: 'Modern hip-hop production template with 808s and samples',
    genre: 'Hip-Hop',
    trackCount: 4,
    duration: '3:00',
    isOfficial: true,
  },
]

export function CreateProjectModal({ isOpen, onClose, onCreate }: CreateProjectModalProps) {
  const [step, setStep] = useState<'template' | 'details'>('template')
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    isPublic: false,
    tags: [] as string[],
  })
  const [newTag, setNewTag] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  const tagInputRef = useRef<HTMLInputElement>(null)

  const commonTags = ['electronic', 'pop', 'jazz', 'classical', 'ambient', 'rock', 'hip-hop', 'experimental']

  const resetForm = () => {
    setStep('template')
    setSelectedTemplate(null)
    setFormData({
      name: '',
      description: '',
      isPublic: false,
      tags: [],
    })
    setNewTag('')
    setErrors({})
  }

  const handleClose = () => {
    if (!isCreating) {
      resetForm()
      onClose()
    }
  }

  const handleTemplateSelect = (templateId: string | null) => {
    setSelectedTemplate(templateId)
    
    if (templateId) {
      const template = projectTemplates.find(t => t.id === templateId)
      if (template) {
        setFormData(prev => ({
          ...prev,
          name: prev.name || `My ${template.name}`,
          tags: prev.tags.length === 0 ? [template.genre.toLowerCase()] : prev.tags,
        }))
      }
    }
  }

  const handleNext = () => {
    if (step === 'template') {
      setStep('details')
    }
  }

  const handleBack = () => {
    if (step === 'details') {
      setStep('template')
    }
  }

  const addTag = (tag: string) => {
    const trimmedTag = tag.trim().toLowerCase()
    if (trimmedTag && !formData.tags.includes(trimmedTag)) {
      setFormData(prev => ({
        ...prev,
        tags: [...prev.tags, trimmedTag],
      }))
    }
    setNewTag('')
  }

  const removeTag = (tagToRemove: string) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove),
    }))
  }

  const handleTagInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault()
      addTag(newTag)
    }
  }

  const validateForm = () => {
    const newErrors: Record<string, string> = {}

    if (!formData.name.trim()) {
      newErrors.name = 'Project name is required'
    } else if (formData.name.length < 3) {
      newErrors.name = 'Project name must be at least 3 characters'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleCreate = async () => {
    if (!validateForm()) return

    setIsCreating(true)
    try {
      await onCreate({
        name: formData.name,
        description: formData.description || undefined,
        isPublic: formData.isPublic,
        tags: formData.tags,
        templateId: selectedTemplate || undefined,
      })
      
      resetForm()
      onClose()
    } catch (error) {
      console.error('Failed to create project:', error)
      setErrors({ general: 'Failed to create project. Please try again.' })
    } finally {
      setIsCreating(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={handleClose}
        className="absolute inset-0 bg-black bg-opacity-50"
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="relative bg-gray-800 rounded-2xl border border-gray-700 w-full max-w-4xl max-h-[90vh] overflow-hidden shadow-2xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-2xl font-bold text-white">Create New Project</h2>
            <p className="text-gray-400 mt-1">
              {step === 'template' ? 'Choose a template to get started' : 'Configure your project settings'}
            </p>
          </div>
          <button
            onClick={handleClose}
            disabled={isCreating}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          <AnimatePresence mode="wait">
            {step === 'template' && (
              <motion.div
                key="template"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="space-y-6"
              >
                {/* Template Selection */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Choose a Template</h3>
                  
                  {/* Blank Project Option */}
                  <div
                    onClick={() => handleTemplateSelect(null)}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all mb-4 ${
                      selectedTemplate === null
                        ? 'border-purple-500 bg-purple-900/20'
                        : 'border-gray-600 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center space-x-4">
                      <div className="w-16 h-16 bg-gray-700 rounded-lg flex items-center justify-center">
                        <FolderIcon className="w-8 h-8 text-gray-400" />
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-white">Blank Project</h4>
                        <p className="text-gray-400 text-sm">Start from scratch with an empty project</p>
                      </div>
                      {selectedTemplate === null && (
                        <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center">
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="w-3 h-3 bg-white rounded-full"
                          />
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Template Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {projectTemplates.map((template) => (
                      <Card
                        key={template.id}
                        onClick={() => handleTemplateSelect(template.id)}
                        className={`cursor-pointer transition-all border-2 ${
                          selectedTemplate === template.id
                            ? 'border-purple-500 bg-purple-900/20'
                            : 'border-gray-600 hover:border-gray-500 bg-gray-800'
                        }`}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start space-x-4">
                            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center flex-shrink-0">
                              <MusicalNoteIcon className="w-6 h-6 text-white" />
                            </div>
                            
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-2 mb-1">
                                <h4 className="font-semibold text-white truncate">{template.name}</h4>
                                {template.isOfficial && (
                                  <SparklesIcon className="w-4 h-4 text-yellow-400 flex-shrink-0" />
                                )}
                              </div>
                              <p className="text-gray-400 text-sm line-clamp-2 mb-2">
                                {template.description}
                              </p>
                              
                              <div className="flex items-center space-x-4 text-xs text-gray-500">
                                <span>{template.genre}</span>
                                <span>{template.trackCount} tracks</span>
                                <span>{template.duration}</span>
                              </div>
                            </div>

                            {selectedTemplate === template.id && (
                              <div className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
                                <motion.div
                                  initial={{ scale: 0 }}
                                  animate={{ scale: 1 }}
                                  className="w-3 h-3 bg-white rounded-full"
                                />
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {step === 'details' && (
              <motion.div
                key="details"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="space-y-6"
              >
                {/* Project Details */}
                <div>
                  <h3 className="text-lg font-semibold text-white mb-4">Project Details</h3>
                  
                  <div className="space-y-4">
                    {/* Project Name */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Project Name *
                      </label>
                      <Input
                        value={formData.name}
                        onChange={(e) => {
                          setFormData(prev => ({ ...prev, name: e.target.value }))
                          if (errors.name) {
                            setErrors(prev => ({ ...prev, name: '' }))
                          }
                        }}
                        placeholder="Enter project name..."
                        className={`bg-gray-700 border-gray-600 text-white ${
                          errors.name ? 'border-red-500' : ''
                        }`}
                      />
                      {errors.name && (
                        <p className="text-red-400 text-sm mt-1">{errors.name}</p>
                      )}
                    </div>

                    {/* Description */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Description
                      </label>
                      <Textarea
                        value={formData.description}
                        onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                        placeholder="Describe your project..."
                        rows={3}
                        className="bg-gray-700 border-gray-600 text-white"
                      />
                    </div>

                    {/* Privacy */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-3">
                        Privacy
                      </label>
                      <div className="space-y-3">
                        <label className="flex items-center space-x-3 cursor-pointer">
                          <input
                            type="radio"
                            checked={!formData.isPublic}
                            onChange={() => setFormData(prev => ({ ...prev, isPublic: false }))}
                            className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 focus:ring-purple-500"
                          />
                          <div className="flex items-center space-x-2">
                            <LockClosedIcon className="w-4 h-4 text-yellow-400" />
                            <span className="text-white">Private</span>
                          </div>
                          <span className="text-gray-400 text-sm">Only you and collaborators can access</span>
                        </label>
                        
                        <label className="flex items-center space-x-3 cursor-pointer">
                          <input
                            type="radio"
                            checked={formData.isPublic}
                            onChange={() => setFormData(prev => ({ ...prev, isPublic: true }))}
                            className="w-4 h-4 text-purple-600 bg-gray-700 border-gray-600 focus:ring-purple-500"
                          />
                          <div className="flex items-center space-x-2">
                            <GlobeAltIcon className="w-4 h-4 text-green-400" />
                            <span className="text-white">Public</span>
                          </div>
                          <span className="text-gray-400 text-sm">Anyone can discover and listen</span>
                        </label>
                      </div>
                    </div>

                    {/* Tags */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Tags
                      </label>
                      
                      {/* Tag Input */}
                      <div className="flex space-x-2 mb-3">
                        <Input
                          ref={tagInputRef}
                          value={newTag}
                          onChange={(e) => setNewTag(e.target.value)}
                          onKeyDown={handleTagInputKeyDown}
                          placeholder="Add tags..."
                          className="bg-gray-700 border-gray-600 text-white flex-1"
                        />
                        <Button
                          type="button"
                          onClick={() => addTag(newTag)}
                          disabled={!newTag.trim()}
                          variant="outline"
                          className="border-gray-600 text-gray-300 hover:bg-gray-700"
                        >
                          <TagIcon className="w-4 h-4" />
                        </Button>
                      </div>

                      {/* Common Tags */}
                      <div className="mb-3">
                        <p className="text-xs text-gray-400 mb-2">Popular tags:</p>
                        <div className="flex flex-wrap gap-2">
                          {commonTags.map((tag) => (
                            <button
                              key={tag}
                              type="button"
                              onClick={() => addTag(tag)}
                              disabled={formData.tags.includes(tag)}
                              className={`px-2 py-1 rounded-full text-xs transition-colors ${
                                formData.tags.includes(tag)
                                  ? 'bg-purple-600 text-white cursor-not-allowed'
                                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                              }`}
                            >
                              {tag}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Selected Tags */}
                      {formData.tags.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                          {formData.tags.map((tag) => (
                            <span
                              key={tag}
                              className="inline-flex items-center space-x-1 px-3 py-1 bg-purple-600 text-white text-sm rounded-full"
                            >
                              <span>{tag}</span>
                              <button
                                type="button"
                                onClick={() => removeTag(tag)}
                                className="hover:text-red-300 transition-colors"
                              >
                                <XMarkIcon className="w-3 h-3" />
                              </button>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Selected Template Summary */}
                {selectedTemplate && (
                  <div className="border-t border-gray-700 pt-6">
                    <h4 className="text-sm font-medium text-gray-300 mb-3">Selected Template</h4>
                    <div className="bg-gray-700 rounded-lg p-4">
                      {(() => {
                        const template = projectTemplates.find(t => t.id === selectedTemplate)
                        return template ? (
                          <div className="flex items-center space-x-3">
                            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                              <MusicalNoteIcon className="w-5 h-5 text-white" />
                            </div>
                            <div>
                              <p className="text-white font-medium">{template.name}</p>
                              <p className="text-gray-400 text-sm">{template.description}</p>
                            </div>
                          </div>
                        ) : null
                      })()}
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {errors.general && (
                  <div className="bg-red-900/50 border border-red-700 rounded-lg p-3">
                    <p className="text-red-200 text-sm">{errors.general}</p>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-700">
          <div className="flex items-center space-x-2">
            {step === 'details' && (
              <Button
                onClick={handleBack}
                variant="ghost"
                disabled={isCreating}
                className="text-gray-400 hover:text-white"
              >
                Back
              </Button>
            )}
          </div>

          <div className="flex items-center space-x-3">
            <Button
              onClick={handleClose}
              variant="ghost"
              disabled={isCreating}
              className="text-gray-400 hover:text-white"
            >
              Cancel
            </Button>
            
            {step === 'template' ? (
              <Button
                onClick={handleNext}
                className="bg-purple-600 hover:bg-purple-700"
              >
                Next
              </Button>
            ) : (
              <Button
                onClick={handleCreate}
                disabled={isCreating || !formData.name.trim()}
                className="bg-purple-600 hover:bg-purple-700"
              >
                {isCreating ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Creating...</span>
                  </div>
                ) : (
                  'Create Project'
                )}
              </Button>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  )
}