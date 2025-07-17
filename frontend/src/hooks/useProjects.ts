'use client'

import { useState, useEffect, useCallback } from 'react'
import { Project, ProjectTrack, ProjectComment, ProjectVersion, projectService } from '@/lib/projects'
import { useAuth } from '@/contexts/AuthContext'
import { useAnalytics } from '@/components/analytics/AnalyticsProvider'

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 20,
    total: 0,
  })

  const { user } = useAuth()
  const { track } = useAnalytics()

  // Fetch projects
  const fetchProjects = useCallback(async (params?: {
    page?: number
    limit?: number
    sort?: 'name' | 'created' | 'updated'
    order?: 'asc' | 'desc'
    search?: string
    tags?: string[]
  }) => {
    if (!user) return

    setIsLoading(true)
    setError(null)

    try {
      const result = await projectService.getUserProjects(params)
      setProjects(result.projects)
      setPagination({
        page: result.page,
        limit: result.limit,
        total: result.total,
      })

      track('projects_fetched', {
        count: result.projects.length,
        page: result.page,
        search: params?.search,
        tags: params?.tags,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch projects')
    } finally {
      setIsLoading(false)
    }
  }, [user, track])

  // Load projects on mount
  useEffect(() => {
    fetchProjects()
  }, [fetchProjects])

  // Create project
  const createProject = useCallback(async (data: {
    name: string
    description?: string
    isPublic?: boolean
    tags?: string[]
    templateId?: string
  }) => {
    try {
      const newProject = await projectService.createProject(data)
      setProjects(prev => [newProject, ...prev])
      
      track('project_created', {
        projectId: newProject.id,
        name: data.name,
        isPublic: data.isPublic,
        hasTemplate: !!data.templateId,
        tagCount: data.tags?.length || 0,
      })

      return newProject
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create project')
      throw err
    }
  }, [track])

  // Update project
  const updateProject = useCallback(async (id: string, updates: Partial<Project>) => {
    try {
      const updatedProject = await projectService.updateProject(id, updates)
      setProjects(prev => prev.map(p => p.id === id ? updatedProject : p))
      
      track('project_updated', {
        projectId: id,
        fields: Object.keys(updates),
      })

      return updatedProject
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update project')
      throw err
    }
  }, [track])

  // Delete project
  const deleteProject = useCallback(async (id: string) => {
    try {
      await projectService.deleteProject(id)
      setProjects(prev => prev.filter(p => p.id !== id))
      
      track('project_deleted', { projectId: id })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete project')
      throw err
    }
  }, [track])

  // Duplicate project
  const duplicateProject = useCallback(async (id: string, name: string) => {
    try {
      const duplicatedProject = await projectService.duplicateProject(id, name)
      setProjects(prev => [duplicatedProject, ...prev])
      
      track('project_duplicated', {
        originalId: id,
        duplicatedId: duplicatedProject.id,
        name,
      })

      return duplicatedProject
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to duplicate project')
      throw err
    }
  }, [track])

  // Search projects
  const searchProjects = useCallback(async (query: string) => {
    await fetchProjects({ 
      search: query,
      page: 1,
    })
  }, [fetchProjects])

  // Filter projects by tags
  const filterByTags = useCallback(async (tags: string[]) => {
    await fetchProjects({ 
      tags,
      page: 1,
    })
  }, [fetchProjects])

  // Load more projects (pagination)
  const loadMore = useCallback(async () => {
    if (pagination.page * pagination.limit >= pagination.total) return

    const nextPage = pagination.page + 1
    await fetchProjects({ page: nextPage })
  }, [fetchProjects, pagination])

  return {
    projects,
    isLoading,
    error,
    pagination,
    fetchProjects,
    createProject,
    updateProject,
    deleteProject,
    duplicateProject,
    searchProjects,
    filterByTags,
    loadMore,
    hasMore: pagination.page * pagination.limit < pagination.total,
  }
}

export function useProject(projectId: string | null) {
  const [project, setProject] = useState<Project | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [autoSaveStatus, setAutoSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')

  const { track } = useAnalytics()

  // Auto-save timer
  const [autoSaveTimer, setAutoSaveTimer] = useState<NodeJS.Timeout | null>(null)

  // Fetch project
  const fetchProject = useCallback(async () => {
    if (!projectId) return

    setIsLoading(true)
    setError(null)

    try {
      const fetchedProject = await projectService.getProject(projectId)
      setProject(fetchedProject)

      track('project_opened', {
        projectId,
        trackCount: fetchedProject.tracks.length,
        collaboratorCount: fetchedProject.collaborators.length,
        isPublic: fetchedProject.isPublic,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch project')
    } finally {
      setIsLoading(false)
    }
  }, [projectId, track])

  // Load project on mount or ID change
  useEffect(() => {
    fetchProject()
  }, [fetchProject])

  // Auto-save function
  const autoSave = useCallback(async (changes: Partial<Project>) => {
    if (!projectId || !project?.settings.autoSave) return

    setAutoSaveStatus('saving')
    
    try {
      await projectService.autoSave(projectId, changes)
      setAutoSaveStatus('saved')
      
      // Reset to idle after 2 seconds
      setTimeout(() => setAutoSaveStatus('idle'), 2000)
    } catch (err) {
      setAutoSaveStatus('error')
    }
  }, [projectId, project?.settings.autoSave])

  // Update project with auto-save
  const updateProjectWithAutoSave = useCallback((updates: Partial<Project>) => {
    if (!project) return

    const updatedProject = { ...project, ...updates }
    setProject(updatedProject)

    // Clear existing timer
    if (autoSaveTimer) {
      clearTimeout(autoSaveTimer)
    }

    // Set new timer for auto-save
    if (project.settings.autoSave) {
      const timer = setTimeout(() => {
        autoSave(updates)
      }, project.settings.saveInterval * 1000) // Convert to milliseconds

      setAutoSaveTimer(timer)
    }
  }, [project, autoSave, autoSaveTimer])

  // Add track
  const addTrack = useCallback(async (trackData: {
    name: string
    audioUrl: string
    generationParams: any
    position?: number
  }) => {
    if (!projectId) return

    try {
      const newTrack = await projectService.addTrack(projectId, trackData)
      setProject(prev => prev ? {
        ...prev,
        tracks: [...prev.tracks, newTrack],
        metadata: {
          ...prev.metadata,
          trackCount: prev.tracks.length + 1,
        }
      } : null)

      track('project_track_added', {
        projectId,
        trackId: newTrack.id,
        trackName: trackData.name,
        position: trackData.position,
      })

      return newTrack
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add track')
      throw err
    }
  }, [projectId, track])

  // Update track
  const updateTrack = useCallback(async (trackId: string, updates: Partial<ProjectTrack>) => {
    if (!projectId) return

    try {
      const updatedTrack = await projectService.updateTrack(projectId, trackId, updates)
      setProject(prev => prev ? {
        ...prev,
        tracks: prev.tracks.map(t => t.id === trackId ? updatedTrack : t),
      } : null)

      track('project_track_updated', {
        projectId,
        trackId,
        fields: Object.keys(updates),
      })

      return updatedTrack
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update track')
      throw err
    }
  }, [projectId, track])

  // Remove track
  const removeTrack = useCallback(async (trackId: string) => {
    if (!projectId) return

    try {
      await projectService.removeTrack(projectId, trackId)
      setProject(prev => prev ? {
        ...prev,
        tracks: prev.tracks.filter(t => t.id !== trackId),
        metadata: {
          ...prev.metadata,
          trackCount: prev.tracks.length - 1,
        }
      } : null)

      track('project_track_removed', {
        projectId,
        trackId,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove track')
      throw err
    }
  }, [projectId, track])

  // Export project
  const exportProject = useCallback(async (format: 'wav' | 'mp3' | 'flac' | 'midi' | 'stems') => {
    if (!projectId) return

    try {
      const result = await projectService.exportProject(projectId, format)
      
      track('project_exported', {
        projectId,
        format,
        filename: result.filename,
      })

      // Trigger download
      const link = document.createElement('a')
      link.href = result.downloadUrl
      link.download = result.filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      return result
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export project')
      throw err
    }
  }, [projectId, track])

  // Cleanup auto-save timer
  useEffect(() => {
    return () => {
      if (autoSaveTimer) {
        clearTimeout(autoSaveTimer)
      }
    }
  }, [autoSaveTimer])

  return {
    project,
    isLoading,
    error,
    autoSaveStatus,
    fetchProject,
    updateProject: updateProjectWithAutoSave,
    addTrack,
    updateTrack,
    removeTrack,
    exportProject,
  }
}

export function useProjectVersions(projectId: string | null) {
  const [versions, setVersions] = useState<ProjectVersion[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { track } = useAnalytics()

  // Fetch versions
  const fetchVersions = useCallback(async () => {
    if (!projectId) return

    setIsLoading(true)
    setError(null)

    try {
      const fetchedVersions = await projectService.getProjectVersions(projectId)
      setVersions(fetchedVersions)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch versions')
    } finally {
      setIsLoading(false)
    }
  }, [projectId])

  // Create version
  const createVersion = useCallback(async (data: {
    name: string
    description?: string
  }) => {
    if (!projectId) return

    try {
      const newVersion = await projectService.createVersion(projectId, data)
      setVersions(prev => [newVersion, ...prev])
      
      track('project_version_created', {
        projectId,
        versionId: newVersion.id,
        versionName: data.name,
      })

      return newVersion
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create version')
      throw err
    }
  }, [projectId, track])

  // Restore version
  const restoreVersion = useCallback(async (versionId: string) => {
    if (!projectId) return

    try {
      const restoredProject = await projectService.restoreVersion(projectId, versionId)
      
      track('project_version_restored', {
        projectId,
        versionId,
      })

      return restoredProject
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restore version')
      throw err
    }
  }, [projectId, track])

  // Load versions on mount
  useEffect(() => {
    fetchVersions()
  }, [fetchVersions])

  return {
    versions,
    isLoading,
    error,
    fetchVersions,
    createVersion,
    restoreVersion,
  }
}

export function useProjectComments(projectId: string | null) {
  const [comments, setComments] = useState<ProjectComment[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { track } = useAnalytics()

  // Fetch comments
  const fetchComments = useCallback(async () => {
    if (!projectId) return

    setIsLoading(true)
    setError(null)

    try {
      const fetchedComments = await projectService.getComments(projectId)
      setComments(fetchedComments)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch comments')
    } finally {
      setIsLoading(false)
    }
  }, [projectId])

  // Add comment
  const addComment = useCallback(async (data: {
    content: string
    trackId?: string
    timestamp?: number
    position?: number
  }) => {
    if (!projectId) return

    try {
      const newComment = await projectService.addComment(projectId, data)
      setComments(prev => [newComment, ...prev])
      
      track('project_comment_added', {
        projectId,
        commentId: newComment.id,
        hasTrackReference: !!data.trackId,
        hasTimestamp: !!data.timestamp,
      })

      return newComment
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add comment')
      throw err
    }
  }, [projectId, track])

  // Load comments on mount
  useEffect(() => {
    fetchComments()
  }, [fetchComments])

  return {
    comments,
    isLoading,
    error,
    fetchComments,
    addComment,
  }
}