'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FolderIcon,
  PlusIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  EllipsisVerticalIcon,
  PlayIcon,
  ShareIcon,
  DocumentDuplicateIcon,
  TrashIcon,
  CalendarIcon,
  MusicalNoteIcon,
  UserGroupIcon,
  LockClosedIcon,
  GlobeAltIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { useProjects } from '@/hooks/useProjects'
import { Project } from '@/lib/projects'
import { CreateProjectModal } from './CreateProjectModal'
import { ProjectOptionsMenu } from './ProjectOptionsMenu'

export function ProjectManager() {
  const {
    projects,
    isLoading,
    error,
    pagination,
    createProject,
    updateProject,
    deleteProject,
    duplicateProject,
    searchProjects,
    filterByTags,
    loadMore,
    hasMore,
  } = useProjects()

  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'name' | 'created' | 'updated'>('updated')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    searchProjects(query)
  }

  const handleTagFilter = (tags: string[]) => {
    setSelectedTags(tags)
    filterByTags(tags)
  }

  const handleSort = (sort: 'name' | 'created' | 'updated', order: 'asc' | 'desc') => {
    setSortBy(sort)
    setSortOrder(order)
    // In real implementation, this would trigger a re-fetch with sort parameters
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  const commonTags = ['electronic', 'pop', 'jazz', 'classical', 'ambient', 'rock', 'hip-hop', 'experimental']

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error}</p>
          <Button onClick={() => window.location.reload()}>
            Try Again
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">My Projects</h1>
          <p className="text-gray-400">
            {projects.length} project{projects.length !== 1 ? 's' : ''} • {pagination.total} total
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <Button
            onClick={() => setShowCreateModal(true)}
            className="bg-purple-600 hover:bg-purple-700 flex items-center space-x-2"
          >
            <PlusIcon className="w-5 h-5" />
            <span>New Project</span>
          </Button>
          
          <Button
            variant="outline"
            className="border-gray-600 text-gray-300 hover:bg-gray-700"
            onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
          >
            {viewMode === 'grid' ? 'List View' : 'Grid View'}
          </Button>
        </div>
      </div>

      {/* Filters and Search */}
      <Card className="bg-gray-800 border-gray-700">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search */}
            <div className="flex-1">
              <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <Input
                  placeholder="Search projects..."
                  value={searchQuery}
                  onChange={(e) => handleSearch(e.target.value)}
                  className="pl-10 bg-gray-700 border-gray-600 text-white"
                />
              </div>
            </div>

            {/* Sort */}
            <div className="flex gap-3">
              <Select value={sortBy} onValueChange={(value: any) => handleSort(value, sortOrder)}>
                <SelectTrigger className="w-40 bg-gray-700 border-gray-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="updated">Last Updated</SelectItem>
                  <SelectItem value="created">Created Date</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                </SelectContent>
              </Select>

              <Select value={sortOrder} onValueChange={(value: any) => handleSort(sortBy, value)}>
                <SelectTrigger className="w-32 bg-gray-700 border-gray-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="desc">Newest</SelectItem>
                  <SelectItem value="asc">Oldest</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Tag Filters */}
          <div className="mt-4">
            <div className="flex items-center space-x-2 mb-3">
              <FunnelIcon className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Filter by tags:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {commonTags.map((tag) => (
                <button
                  key={tag}
                  onClick={() => {
                    const newTags = selectedTags.includes(tag)
                      ? selectedTags.filter(t => t !== tag)
                      : [...selectedTags, tag]
                    handleTagFilter(newTags)
                  }}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                    selectedTags.includes(tag)
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {tag}
                </button>
              ))}
              {selectedTags.length > 0 && (
                <button
                  onClick={() => handleTagFilter([])}
                  className="px-3 py-1 rounded-full text-xs font-medium text-red-400 bg-red-900/20 hover:bg-red-900/30"
                >
                  Clear All
                </button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Projects Grid/List */}
      {isLoading && projects.length === 0 ? (
        <div className="flex items-center justify-center min-h-96">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-500"></div>
        </div>
      ) : projects.length === 0 ? (
        <div className="text-center py-12">
          <FolderIcon className="w-24 h-24 text-gray-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">No Projects Found</h3>
          <p className="text-gray-400 mb-6">
            {searchQuery || selectedTags.length > 0
              ? 'Try adjusting your search or filters'
              : 'Create your first project to get started'
            }
          </p>
          {!searchQuery && selectedTags.length === 0 && (
            <Button
              onClick={() => setShowCreateModal(true)}
              className="bg-purple-600 hover:bg-purple-700"
            >
              <PlusIcon className="w-5 h-5 mr-2" />
              Create Project
            </Button>
          )}
        </div>
      ) : (
        <div className={viewMode === 'grid' 
          ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6'
          : 'space-y-4'
        }>
          <AnimatePresence>
            {projects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                viewMode={viewMode}
                onUpdate={updateProject}
                onDelete={deleteProject}
                onDuplicate={duplicateProject}
              />
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Load More */}
      {hasMore && (
        <div className="text-center">
          <Button
            onClick={loadMore}
            disabled={isLoading}
            variant="outline"
            className="border-gray-600 text-gray-300 hover:bg-gray-700"
          >
            {isLoading ? 'Loading...' : 'Load More Projects'}
          </Button>
        </div>
      )}

      {/* Create Project Modal */}
      <CreateProjectModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreate={createProject}
      />
    </div>
  )
}

interface ProjectCardProps {
  project: Project
  viewMode: 'grid' | 'list'
  onUpdate: (id: string, updates: Partial<Project>) => Promise<Project>
  onDelete: (id: string) => Promise<void>
  onDuplicate: (id: string, name: string) => Promise<Project>
}

function ProjectCard({ project, viewMode, onUpdate, onDelete, onDuplicate }: ProjectCardProps) {
  const [showOptions, setShowOptions] = useState(false)

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  if (viewMode === 'list') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="bg-gray-800 border border-gray-700 rounded-lg hover:border-gray-600 transition-colors"
      >
        <div className="p-4 flex items-center space-x-4">
          {/* Project Thumbnail */}
          <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center flex-shrink-0">
            <MusicalNoteIcon className="w-8 h-8 text-white" />
          </div>

          {/* Project Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h3 className="font-semibold text-white truncate">{project.name}</h3>
              {project.isPublic ? (
                <GlobeAltIcon className="w-4 h-4 text-green-400" />
              ) : (
                <LockClosedIcon className="w-4 h-4 text-yellow-400" />
              )}
            </div>
            <p className="text-gray-400 text-sm truncate">{project.description}</p>
            
            <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
              <span className="flex items-center space-x-1">
                <MusicalNoteIcon className="w-3 h-3" />
                <span>{project.metadata.trackCount} tracks</span>
              </span>
              <span className="flex items-center space-x-1">
                <UserGroupIcon className="w-3 h-3" />
                <span>{project.collaborators.length} collaborators</span>
              </span>
              <span className="flex items-center space-x-1">
                <CalendarIcon className="w-3 h-3" />
                <span>Updated {formatDate(project.updatedAt)}</span>
              </span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              className="bg-purple-600 hover:bg-purple-700"
            >
              <PlayIcon className="w-4 h-4" />
            </Button>
            
            <div className="relative">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowOptions(!showOptions)}
                className="text-gray-400 hover:text-white"
              >
                <EllipsisVerticalIcon className="w-4 h-4" />
              </Button>
              
              <ProjectOptionsMenu
                project={project}
                isOpen={showOptions}
                onClose={() => setShowOptions(false)}
                onUpdate={onUpdate}
                onDelete={onDelete}
                onDuplicate={onDuplicate}
              />
            </div>
          </div>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      whileHover={{ scale: 1.02 }}
      className="group"
    >
      <Card className="bg-gray-800 border-gray-700 hover:border-gray-600 transition-colors h-full">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2 mb-2">
                <CardTitle className="text-white truncate text-base">{project.name}</CardTitle>
                {project.isPublic ? (
                  <GlobeAltIcon className="w-4 h-4 text-green-400 flex-shrink-0" />
                ) : (
                  <LockClosedIcon className="w-4 h-4 text-yellow-400 flex-shrink-0" />
                )}
              </div>
              <p className="text-gray-400 text-sm line-clamp-2">{project.description}</p>
            </div>
            
            <div className="relative">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowOptions(!showOptions)}
                className="text-gray-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <EllipsisVerticalIcon className="w-4 h-4" />
              </Button>
              
              <ProjectOptionsMenu
                project={project}
                isOpen={showOptions}
                onClose={() => setShowOptions(false)}
                onUpdate={onUpdate}
                onDelete={onDelete}
                onDuplicate={onDuplicate}
              />
            </div>
          </div>
        </CardHeader>

        <CardContent className="pt-0">
          {/* Project Thumbnail */}
          <div className="w-full h-32 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg mb-4 flex items-center justify-center">
            <MusicalNoteIcon className="w-12 h-12 text-white" />
          </div>

          {/* Tags */}
          {project.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-3">
              {project.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded-full"
                >
                  {tag}
                </span>
              ))}
              {project.tags.length > 3 && (
                <span className="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded-full">
                  +{project.tags.length - 3}
                </span>
              )}
            </div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
            <div>
              <div className="text-gray-400">Tracks</div>
              <div className="text-white font-medium">{project.metadata.trackCount}</div>
            </div>
            <div>
              <div className="text-gray-400">Duration</div>
              <div className="text-white font-medium">{formatDuration(project.metadata.totalDuration)}</div>
            </div>
            <div>
              <div className="text-gray-400">Collaborators</div>
              <div className="text-white font-medium">{project.collaborators.length}</div>
            </div>
            <div>
              <div className="text-gray-400">Updated</div>
              <div className="text-white font-medium">{formatDate(project.updatedAt)}</div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex space-x-2">
            <Button
              size="sm"
              className="bg-purple-600 hover:bg-purple-700 flex-1"
            >
              <PlayIcon className="w-4 h-4 mr-2" />
              Open
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="border-gray-600 text-gray-300 hover:bg-gray-700"
            >
              <ShareIcon className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}