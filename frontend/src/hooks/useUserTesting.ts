'use client'

import { useState, useEffect, useCallback, useRef } from 'react'

interface UserAction {
  type: 'click' | 'scroll' | 'hover' | 'input' | 'navigation' | 'error'
  timestamp: number
  element?: string
  page: string
  coordinates?: { x: number; y: number }
  value?: string
  metadata?: Record<string, any>
}

interface UserSession {
  sessionId: string
  userId?: string
  startTime: number
  endTime?: number
  actions: UserAction[]
  device: {
    userAgent: string
    viewport: { width: number; height: number }
    isMobile: boolean
    isTablet: boolean
  }
  performance: {
    pageLoadTime: number
    firstContentfulPaint?: number
    largestContentfulPaint?: number
  }
}

interface TestScenario {
  id: string
  name: string
  description: string
  tasks: Task[]
  successCriteria: string[]
  estimatedDuration: number
}

interface Task {
  id: string
  instruction: string
  expectedActions: string[]
  timeLimit?: number
  helpText?: string
}

interface TestResults {
  scenarioId: string
  userId: string
  startTime: number
  endTime: number
  tasksCompleted: number
  totalTasks: number
  errors: number
  helpRequests: number
  satisfaction: number
  timeSpent: number
  feedback: string
}

export function useUserTesting() {
  const [currentSession, setCurrentSession] = useState<UserSession | null>(null)
  const [isTracking, setIsTracking] = useState(false)
  const [currentScenario, setCurrentScenario] = useState<TestScenario | null>(null)
  const [testResults, setTestResults] = useState<TestResults | null>(null)
  
  const actionsRef = useRef<UserAction[]>([])
  const sessionIdRef = useRef<string>('')

  // Initialize session
  const startSession = useCallback((userId?: string) => {
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    sessionIdRef.current = sessionId
    
    const session: UserSession = {
      sessionId,
      userId,
      startTime: Date.now(),
      actions: [],
      device: {
        userAgent: navigator.userAgent,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
        },
        isMobile: /iPhone|iPad|iPod|Android/i.test(navigator.userAgent),
        isTablet: /iPad|Android/i.test(navigator.userAgent) && window.innerWidth > 768,
      },
      performance: {
        pageLoadTime: performance.now(),
        firstContentfulPaint: 0,
        largestContentfulPaint: 0,
      },
    }

    // Capture performance metrics
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.name === 'first-contentful-paint') {
            session.performance.firstContentfulPaint = entry.startTime
          }
          if (entry.name === 'largest-contentful-paint') {
            session.performance.largestContentfulPaint = entry.startTime
          }
        })
      })
      observer.observe({ entryTypes: ['paint', 'largest-contentful-paint'] })
    }

    setCurrentSession(session)
    actionsRef.current = []
    setIsTracking(true)
    
    return sessionId
  }, [])

  // Track user action
  const trackAction = useCallback((action: Omit<UserAction, 'timestamp' | 'page'>) => {
    if (!isTracking || !currentSession) return

    const fullAction: UserAction = {
      ...action,
      timestamp: Date.now(),
      page: window.location.pathname,
    }

    actionsRef.current.push(fullAction)
    
    setCurrentSession(prev => prev ? {
      ...prev,
      actions: [...prev.actions, fullAction],
    } : null)
  }, [isTracking, currentSession])

  // End session
  const endSession = useCallback(async () => {
    if (!currentSession) return null

    const finalSession: UserSession = {
      ...currentSession,
      endTime: Date.now(),
      actions: actionsRef.current,
    }

    setCurrentSession(finalSession)
    setIsTracking(false)

    // Send session data to analytics endpoint
    try {
      await fetch('/api/analytics/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalSession),
      })
    } catch (error) {
      console.error('Failed to send session data:', error)
    }

    return finalSession
  }, [currentSession])

  // Start test scenario
  const startTestScenario = useCallback((scenario: TestScenario, userId?: string) => {
    const sessionId = startSession(userId)
    setCurrentScenario(scenario)
    
    const results: TestResults = {
      scenarioId: scenario.id,
      userId: userId || 'anonymous',
      startTime: Date.now(),
      endTime: 0,
      tasksCompleted: 0,
      totalTasks: scenario.tasks.length,
      errors: 0,
      helpRequests: 0,
      satisfaction: 0,
      timeSpent: 0,
      feedback: '',
    }
    
    setTestResults(results)
    
    // Track scenario start
    trackAction({
      type: 'navigation',
      element: 'test-scenario-start',
      metadata: { scenarioId: scenario.id, scenarioName: scenario.name },
    })
    
    return sessionId
  }, [startSession, trackAction])

  // Complete test scenario
  const completeTestScenario = useCallback(async (feedback: string, satisfaction: number) => {
    if (!currentScenario || !testResults) return null

    const finalResults: TestResults = {
      ...testResults,
      endTime: Date.now(),
      timeSpent: Date.now() - testResults.startTime,
      feedback,
      satisfaction,
    }

    setTestResults(finalResults)
    
    // Track scenario completion
    trackAction({
      type: 'navigation',
      element: 'test-scenario-complete',
      metadata: { 
        scenarioId: currentScenario.id,
        results: finalResults,
      },
    })

    // Send test results
    try {
      await fetch('/api/analytics/test-results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalResults),
      })
    } catch (error) {
      console.error('Failed to send test results:', error)
    }

    await endSession()
    setCurrentScenario(null)
    
    return finalResults
  }, [currentScenario, testResults, trackAction, endSession])

  // Track specific events
  const trackClick = useCallback((element: string, coordinates?: { x: number; y: number }) => {
    trackAction({
      type: 'click',
      element,
      coordinates,
    })
  }, [trackAction])

  const trackError = useCallback((error: string, element?: string) => {
    trackAction({
      type: 'error',
      element,
      metadata: { error },
    })
    
    if (testResults) {
      setTestResults(prev => prev ? { ...prev, errors: prev.errors + 1 } : null)
    }
  }, [trackAction, testResults])

  const trackTaskCompletion = useCallback((taskId: string) => {
    trackAction({
      type: 'navigation',
      element: 'task-complete',
      metadata: { taskId },
    })
    
    if (testResults) {
      setTestResults(prev => prev ? { 
        ...prev, 
        tasksCompleted: prev.tasksCompleted + 1 
      } : null)
    }
  }, [trackAction, testResults])

  // Auto-track common events
  useEffect(() => {
    if (!isTracking) return

    const handleClick = (event: MouseEvent) => {
      const target = event.target as Element
      const element = target.tagName.toLowerCase()
      const id = target.id
      const className = target.className
      
      trackClick(
        id || `${element}.${className}` || element,
        { x: event.clientX, y: event.clientY }
      )
    }

    const handleScroll = () => {
      trackAction({
        type: 'scroll',
        metadata: {
          scrollTop: window.scrollY,
          scrollLeft: window.scrollX,
        },
      })
    }

    const handleError = (event: ErrorEvent) => {
      trackError(event.message, event.filename)
    }

    document.addEventListener('click', handleClick)
    window.addEventListener('scroll', handleScroll, { passive: true })
    window.addEventListener('error', handleError)

    return () => {
      document.removeEventListener('click', handleClick)
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('error', handleError)
    }
  }, [isTracking, trackClick, trackAction, trackError])

  // Auto-end session on page unload
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (isTracking) {
        endSession()
      }
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => window.removeEventListener('beforeunload', handleBeforeUnload)
  }, [isTracking, endSession])

  return {
    // Session management
    currentSession,
    isTracking,
    startSession,
    endSession,
    
    // Test scenarios
    currentScenario,
    testResults,
    startTestScenario,
    completeTestScenario,
    
    // Action tracking
    trackAction,
    trackClick,
    trackError,
    trackTaskCompletion,
    
    // Utilities
    sessionId: sessionIdRef.current,
    actionsCount: actionsRef.current.length,
  }
}