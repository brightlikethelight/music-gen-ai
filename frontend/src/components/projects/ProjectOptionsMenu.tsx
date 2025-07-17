'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  PencilIcon,
  ShareIcon,
  DocumentDuplicateIcon,
  ArchiveBoxIcon,
  TrashIcon,
  EyeIcon,
  DownloadIcon,
  UserGroupIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline'
import { Project } from '@/lib/projects'

interface ProjectOptionsMenuProps {
  project: Project
  isOpen: boolean
  onClose: () => void
  onUpdate: (id: string, updates: Partial<Project>) => Promise<Project>
  onDelete: (id: string) => Promise<void>
  onDuplicate: (id: string, name: string) => Promise<Project>
}

export function ProjectOptionsMenu({
  project,
  isOpen,
  onClose,
  onUpdate,
  onDelete,
  onDuplicate,
}: ProjectOptionsMenuProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleAction = async (action: string) => {
    setIsProcessing(true)
    
    try {
      switch (action) {
        case 'duplicate':
          await onDuplicate(project.id, `${project.name} (Copy)`)
          break
        case 'archive':
          // In real implementation, add archived status to project
          console.log('Archive project:', project.id)
          break
        case 'delete':
          if (showDeleteConfirm) {
            await onDelete(project.id)
          } else {
            setShowDeleteConfirm(true)
            setIsProcessing(false)
            return
          }
          break
        case 'share':
          // In real implementation, open share modal
          console.log('Share project:', project.id)
          break
        case 'export':
          // In real implementation, trigger export
          console.log('Export project:', project.id)
          break
        case 'settings':
          // In real implementation, open project settings
          console.log('Open project settings:', project.id)
          break
        case 'collaborators':
          // In real implementation, open collaborators modal
          console.log('Manage collaborators:', project.id)
          break
      }
      
      onClose()
      setShowDeleteConfirm(false)
    } catch (error) {
      console.error('Action failed:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  const menuItems = [
    {
      id: 'view',
      label: 'View Details',
      icon: EyeIcon,
      onClick: () => console.log('View project details'),
    },
    {
      id: 'edit',
      label: 'Edit Info',
      icon: PencilIcon,
      onClick: () => console.log('Edit project info'),
    },
    {
      id: 'share',
      label: 'Share',
      icon: ShareIcon,
      onClick: () => handleAction('share'),
    },
    {
      id: 'duplicate',
      label: 'Duplicate',
      icon: DocumentDuplicateIcon,
      onClick: () => handleAction('duplicate'),
    },
    {
      id: 'export',
      label: 'Export',
      icon: DownloadIcon,
      onClick: () => handleAction('export'),
    },
    {
      id: 'collaborators',
      label: 'Collaborators',
      icon: UserGroupIcon,
      onClick: () => handleAction('collaborators'),
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: Cog6ToothIcon,
      onClick: () => handleAction('settings'),
    },
    {
      id: 'divider',
      label: '',
      icon: null,
      onClick: () => {},
    },
    {
      id: 'archive',
      label: 'Archive',
      icon: ArchiveBoxIcon,
      onClick: () => handleAction('archive'),
      variant: 'warning' as const,
    },
    {
      id: 'delete',
      label: showDeleteConfirm ? 'Confirm Delete' : 'Delete',
      icon: TrashIcon,
      onClick: () => handleAction('delete'),
      variant: 'danger' as const,
    },
  ]

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40"
        onClick={onClose}
      />

      {/* Menu */}
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: -10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: -10 }}
          className="absolute right-0 top-8 z-50 w-48 bg-gray-800 border border-gray-700 rounded-lg shadow-xl py-2"
        >
          {menuItems.map((item) => {
            if (item.id === 'divider') {
              return <div key="divider" className="h-px bg-gray-700 my-2" />
            }

            const Icon = item.icon!
            const isDelete = item.id === 'delete'
            const isConfirming = isDelete && showDeleteConfirm

            return (
              <button
                key={item.id}
                onClick={item.onClick}
                disabled={isProcessing}
                className={`w-full flex items-center space-x-3 px-4 py-2 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  item.variant === 'danger'
                    ? isConfirming
                      ? 'text-red-400 bg-red-900/50 hover:bg-red-900/70'
                      : 'text-red-400 hover:bg-red-900/30'
                    : item.variant === 'warning'
                    ? 'text-yellow-400 hover:bg-yellow-900/30'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                <Icon className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">{item.label}</span>
                {isProcessing && item.id === 'delete' && isConfirming && (
                  <div className="w-3 h-3 border border-red-400 border-t-transparent rounded-full animate-spin ml-auto" />
                )}
                {isProcessing && item.id === 'duplicate' && (
                  <div className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin ml-auto" />
                )}
              </button>
            )
          })}

          {showDeleteConfirm && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="px-4 py-2 border-t border-gray-700 mt-2"
            >
              <p className="text-xs text-red-300 mb-2">
                This action cannot be undone. All tracks and data will be permanently deleted.
              </p>
              <div className="flex space-x-2">
                <button
                  onClick={() => setShowDeleteConfirm(false)}
                  disabled={isProcessing}
                  className="flex-1 px-2 py-1 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleAction('delete')}
                  disabled={isProcessing}
                  className="flex-1 px-2 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 transition-colors disabled:opacity-50"
                >
                  {isProcessing ? 'Deleting...' : 'Delete'}
                </button>
              </div>
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>
    </>
  )
}