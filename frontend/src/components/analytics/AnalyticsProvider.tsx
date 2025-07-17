'use client'

import { createContext, useContext, useEffect, ReactNode } from 'react'
import { useUserTesting } from '@/hooks/useUserTesting'

interface AnalyticsConfig {
  enableUserTesting: boolean
  enableErrorTracking: boolean
  enablePerformanceTracking: boolean
  sampleRate: number
  debug: boolean
}

interface AnalyticsContextType {
  track: (event: string, properties?: Record<string, any>) => void
  identify: (userId: string, traits?: Record<string, any>) => void
  page: (name: string, properties?: Record<string, any>) => void
  error: (error: Error, context?: Record<string, any>) => void
  performance: (metric: string, value: number, unit?: string) => void
}

const AnalyticsContext = createContext<AnalyticsContextType | null>(null)

interface AnalyticsProviderProps {
  children: ReactNode
  config?: Partial<AnalyticsConfig>
}

const defaultConfig: AnalyticsConfig = {
  enableUserTesting: true,
  enableErrorTracking: true,
  enablePerformanceTracking: true,
  sampleRate: 1.0,
  debug: process.env.NODE_ENV === 'development',
}

export function AnalyticsProvider({ children, config = {} }: AnalyticsProviderProps) {
  const finalConfig = { ...defaultConfig, ...config }
  const { trackAction, trackError, startSession } = useUserTesting()

  useEffect(() => {
    if (finalConfig.enableUserTesting) {
      // Start tracking session
      startSession()
    }

    // Set up global error tracking
    if (finalConfig.enableErrorTracking) {
      const handleError = (event: ErrorEvent) => {
        trackError(event.message, event.filename)
        
        // Send to external analytics service
        sendAnalyticsEvent('error', {
          message: event.message,
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno,
          stack: event.error?.stack,
        })
      }

      const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
        trackError('Unhandled Promise Rejection', 'promise')
        
        sendAnalyticsEvent('unhandled_rejection', {
          reason: event.reason,
          promise: event.promise,
        })
      }

      window.addEventListener('error', handleError)
      window.addEventListener('unhandledrejection', handleUnhandledRejection)

      return () => {
        window.removeEventListener('error', handleError)
        window.removeEventListener('unhandledrejection', handleUnhandledRejection)
      }
    }
  }, [finalConfig, startSession, trackError])

  // Performance monitoring
  useEffect(() => {
    if (!finalConfig.enablePerformanceTracking) return

    const observer = new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        // Track Core Web Vitals
        if (entry.entryType === 'paint') {
          sendAnalyticsEvent('performance', {
            metric: entry.name,
            value: entry.startTime,
            unit: 'ms',
          })
        }

        if (entry.entryType === 'largest-contentful-paint') {
          sendAnalyticsEvent('performance', {
            metric: 'largest-contentful-paint',
            value: entry.startTime,
            unit: 'ms',
          })
        }

        if (entry.entryType === 'first-input') {
          sendAnalyticsEvent('performance', {
            metric: 'first-input-delay',
            value: (entry as PerformanceEventTiming).processingStart - entry.startTime,
            unit: 'ms',
          })
        }

        if (entry.entryType === 'layout-shift') {
          if (!(entry as any).hadRecentInput) {
            sendAnalyticsEvent('performance', {
              metric: 'cumulative-layout-shift',
              value: (entry as any).value,
              unit: 'score',
            })
          }
        }
      })
    })

    // Observe different performance entry types
    try {
      observer.observe({ entryTypes: ['paint', 'largest-contentful-paint', 'first-input', 'layout-shift'] })
    } catch (error) {
      console.warn('Performance observer not supported:', error)
    }

    return () => observer.disconnect()
  }, [finalConfig.enablePerformanceTracking])

  const sendAnalyticsEvent = async (event: string, properties: Record<string, any>) => {
    // Sample based on configured rate
    if (Math.random() > finalConfig.sampleRate) return

    if (finalConfig.debug) {
      console.log('Analytics Event:', event, properties)
    }

    try {
      // Send to your analytics endpoint
      await fetch('/api/analytics/events', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          event,
          properties: {
            ...properties,
            timestamp: Date.now(),
            url: window.location.href,
            referrer: document.referrer,
            userAgent: navigator.userAgent,
            viewport: {
              width: window.innerWidth,
              height: window.innerHeight,
            },
          },
        }),
      })
    } catch (error) {
      if (finalConfig.debug) {
        console.error('Failed to send analytics event:', error)
      }
    }
  }

  const track = (event: string, properties?: Record<string, any>) => {
    // Track in user testing system
    trackAction({
      type: 'navigation',
      element: event,
      metadata: properties,
    })

    // Send to analytics
    sendAnalyticsEvent('track', { event, ...properties })
  }

  const identify = (userId: string, traits?: Record<string, any>) => {
    sendAnalyticsEvent('identify', { userId, traits })
  }

  const page = (name: string, properties?: Record<string, any>) => {
    trackAction({
      type: 'navigation',
      element: 'page_view',
      metadata: { page: name, ...properties },
    })

    sendAnalyticsEvent('page', { name, ...properties })
  }

  const error = (error: Error, context?: Record<string, any>) => {
    trackError(error.message, error.stack || 'unknown')
    
    sendAnalyticsEvent('error', {
      message: error.message,
      stack: error.stack,
      name: error.name,
      ...context,
    })
  }

  const performance = (metric: string, value: number, unit?: string) => {
    sendAnalyticsEvent('performance', { metric, value, unit })
  }

  const analyticsApi: AnalyticsContextType = {
    track,
    identify,
    page,
    error,
    performance,
  }

  return (
    <AnalyticsContext.Provider value={analyticsApi}>
      {children}
    </AnalyticsContext.Provider>
  )
}

export function useAnalytics() {
  const context = useContext(AnalyticsContext)
  if (!context) {
    throw new Error('useAnalytics must be used within an AnalyticsProvider')
  }
  return context
}

// Helper hooks for specific tracking scenarios
export function usePageTracking(pageName: string) {
  const { page } = useAnalytics()
  
  useEffect(() => {
    page(pageName)
  }, [page, pageName])
}

export function useErrorBoundary() {
  const { error } = useAnalytics()
  
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      error(new Error(event.message), {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      })
    }

    window.addEventListener('error', handleError)
    return () => window.removeEventListener('error', handleError)
  }, [error])
}

// Performance measurement hook
export function usePerformanceMeasurement() {
  const { performance: trackPerformance } = useAnalytics()
  
  const measureFunction = <T extends any[], R>(
    name: string,
    fn: (...args: T) => R
  ) => {
    return (...args: T): R => {
      const start = performance.now()
      const result = fn(...args)
      const duration = performance.now() - start
      
      trackPerformance(`function_${name}`, duration, 'ms')
      return result
    }
  }

  const measureAsync = async <T extends any[], R>(
    name: string,
    fn: (...args: T) => Promise<R>,
    ...args: T
  ): Promise<R> => {
    const start = performance.now()
    const result = await fn(...args)
    const duration = performance.now() - start
    
    trackPerformance(`async_${name}`, duration, 'ms')
    return result
  }

  return { measureFunction, measureAsync }
}