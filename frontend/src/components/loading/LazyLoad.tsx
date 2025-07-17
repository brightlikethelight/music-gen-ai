'use client'

import { Suspense, lazy, ComponentType, ReactNode, useState, useEffect } from 'react'
import { ErrorBoundary } from '../errors/ErrorBoundary'
import { LoadingSpinner } from './LoadingSpinner'
import { ErrorMessage } from '../errors/ErrorMessage'

interface LazyLoadProps {
  loader: () => Promise<{ default: ComponentType<any> }>
  fallback?: ReactNode
  errorFallback?: ReactNode
  delay?: number
  onError?: (error: Error) => void
  props?: Record<string, any>
}

export function LazyLoad({
  loader,
  fallback,
  errorFallback,
  delay = 300,
  onError,
  props = {},
}: LazyLoadProps) {
  const [showFallback, setShowFallback] = useState(false)
  const [Component, setComponent] = useState<ComponentType<any> | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let timeout: NodeJS.Timeout

    // Show fallback after delay to avoid flash for fast loads
    if (delay > 0) {
      timeout = setTimeout(() => {
        setShowFallback(true)
      }, delay)
    } else {
      setShowFallback(true)
    }

    // Load component
    loader()
      .then((module) => {
        setComponent(() => module.default)
        setError(null)
      })
      .catch((err) => {
        setError(err)
        if (onError) {
          onError(err)
        }
      })
      .finally(() => {
        clearTimeout(timeout)
      })

    return () => {
      clearTimeout(timeout)
    }
  }, [loader, delay, onError])

  if (error) {
    if (errorFallback) {
      return <>{errorFallback}</>
    }

    return (
      <ErrorMessage
        severity="error"
        title="Failed to load component"
        message={error.message}
        dismissible={false}
      />
    )
  }

  if (!Component) {
    if (!showFallback) {
      return null
    }

    if (fallback) {
      return <>{fallback}</>
    }

    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  return (
    <ErrorBoundary
      fallback={(error, reset) => (
        errorFallback || (
          <ErrorMessage
            severity="error"
            title="Component Error"
            message={error.message}
            onDismiss={reset}
          />
        )
      )}
    >
      <Component {...props} />
    </ErrorBoundary>
  )
}

// Helper function for creating lazy components with loading states
export function createLazyComponent<T extends ComponentType<any>>(
  loader: () => Promise<{ default: T }>,
  options?: {
    fallback?: ReactNode
    errorFallback?: ReactNode
    delay?: number
  }
) {
  const LazyComponent = lazy(loader)

  return (props: any) => (
    <Suspense
      fallback={
        options?.fallback || (
          <div className="flex items-center justify-center py-8">
            <LoadingSpinner size="lg" />
          </div>
        )
      }
    >
      <ErrorBoundary
        fallback={(error, reset) => (
          options?.errorFallback || (
            <ErrorMessage
              severity="error"
              title="Component Error"
              message={error.message}
              onDismiss={reset}
            />
          )
        )}
      >
        <LazyComponent {...props} />
      </ErrorBoundary>
    </Suspense>
  )
}