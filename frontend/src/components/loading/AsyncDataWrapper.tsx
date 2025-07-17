'use client'

import { ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LoadingSpinner } from './LoadingSpinner'
import { SkeletonLoader } from './SkeletonLoader'
import { ErrorMessage } from '../errors/ErrorMessage'

interface AsyncDataWrapperProps<T> {
  data: T | null | undefined
  isLoading: boolean
  error?: Error | string | null
  loadingComponent?: ReactNode
  errorComponent?: ReactNode
  emptyComponent?: ReactNode
  children: (data: T) => ReactNode
  showSkeleton?: boolean
  skeletonCount?: number
  skeletonVariant?: 'text' | 'circular' | 'rectangular' | 'card' | 'list'
  retryFn?: () => void
  className?: string
}

export function AsyncDataWrapper<T>({
  data,
  isLoading,
  error,
  loadingComponent,
  errorComponent,
  emptyComponent,
  children,
  showSkeleton = false,
  skeletonCount = 3,
  skeletonVariant = 'list',
  retryFn,
  className = '',
}: AsyncDataWrapperProps<T>) {
  if (error) {
    if (errorComponent) {
      return <>{errorComponent}</>
    }

    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className={className}
      >
        <ErrorMessage
          severity="error"
          title="Error Loading Data"
          message={typeof error === 'string' ? error : error.message}
          dismissible={false}
        >
          {retryFn && (
            <button
              onClick={retryFn}
              className="mt-2 px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
            >
              Try Again
            </button>
          )}
        </ErrorMessage>
      </motion.div>
    )
  }

  if (isLoading) {
    if (loadingComponent) {
      return <>{loadingComponent}</>
    }

    if (showSkeleton) {
      return (
        <div className={className}>
          <SkeletonLoader
            variant={skeletonVariant}
            count={skeletonCount}
            animate
          />
        </div>
      )
    }

    return (
      <div className={`flex items-center justify-center py-8 ${className}`}>
        <LoadingSpinner size="lg" text="Loading..." />
      </div>
    )
  }

  if (!data || (Array.isArray(data) && data.length === 0)) {
    if (emptyComponent) {
      return <>{emptyComponent}</>
    }

    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className={`text-center py-8 ${className}`}
      >
        <p className="text-gray-400">No data available</p>
      </motion.div>
    )
  }

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key="content"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={className}
      >
        {children(data)}
      </motion.div>
    </AnimatePresence>
  )
}