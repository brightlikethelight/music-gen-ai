'use client'

import { useState, useEffect, useCallback } from 'react'
import { useAnalytics } from '@/components/analytics/AnalyticsProvider'

export interface ABTest {
  id: string
  name: string
  description: string
  variants: ABVariant[]
  status: 'draft' | 'running' | 'paused' | 'completed'
  startDate: string
  endDate?: string
  trafficAllocation: number // Percentage of users to include in test
  targetMetric: string
  hypothesis: string
  segments?: string[] // User segments to target
}

export interface ABVariant {
  id: string
  name: string
  description: string
  weight: number // Percentage allocation within test
  config: Record<string, any>
  isControl: boolean
}

export interface ABTestResult {
  testId: string
  variantId: string
  metric: string
  value: number
  conversionRate?: number
  significance?: number
  confidenceInterval?: [number, number]
}

interface ABTestState {
  activeTests: Record<string, ABTest>
  userVariants: Record<string, string> // testId -> variantId
  conversions: Record<string, number>
}

export function useABTesting() {
  const [state, setState] = useState<ABTestState>({
    activeTests: {},
    userVariants: {},
    conversions: {},
  })
  const [isLoading, setIsLoading] = useState(true)
  const { track } = useAnalytics()

  // Load active tests and user assignments
  useEffect(() => {
    loadActiveTests()
  }, [])

  const loadActiveTests = async () => {
    try {
      const response = await fetch('/api/ab-testing/active-tests')
      const tests = await response.json()
      
      // Get user's stored variant assignments
      const storedVariants = getStoredVariants()
      
      // Initialize user variants for new tests
      const userVariants: Record<string, string> = { ...storedVariants }
      
      for (const test of tests) {
        if (!userVariants[test.id] && shouldIncludeInTest(test)) {
          userVariants[test.id] = assignVariant(test)
          
          // Track test exposure
          track('ab_test_exposure', {
            testId: test.id,
            variantId: userVariants[test.id],
            testName: test.name,
          })
        }
      }
      
      // Store updated variants
      storeVariants(userVariants)
      
      setState(prev => ({
        ...prev,
        activeTests: tests.reduce((acc: Record<string, ABTest>, test: ABTest) => {
          acc[test.id] = test
          return acc
        }, {}),
        userVariants,
      }))
    } catch (error) {
      console.error('Failed to load A/B tests:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const shouldIncludeInTest = (test: ABTest): boolean => {
    // Check traffic allocation
    const userHash = getUserHash()
    const bucket = userHash % 100
    
    if (bucket >= test.trafficAllocation) {
      return false
    }
    
    // Check segments if specified
    if (test.segments && test.segments.length > 0) {
      // In a real implementation, you'd check user segments
      // For now, we'll include all users
      return true
    }
    
    return true
  }

  const assignVariant = (test: ABTest): string => {
    const userHash = getUserHash()
    const totalWeight = test.variants.reduce((sum, variant) => sum + variant.weight, 0)
    const bucket = userHash % totalWeight
    
    let currentWeight = 0
    for (const variant of test.variants) {
      currentWeight += variant.weight
      if (bucket < currentWeight) {
        return variant.id
      }
    }
    
    // Fallback to control variant
    return test.variants.find(v => v.isControl)?.id || test.variants[0].id
  }

  const getUserHash = (): number => {
    // In a real implementation, use a consistent user ID
    // For demo, we'll use a session-based hash
    const sessionId = sessionStorage.getItem('session-id') || generateSessionId()
    return hashString(sessionId)
  }

  const generateSessionId = (): string => {
    const id = Math.random().toString(36).substr(2, 9)
    sessionStorage.setItem('session-id', id)
    return id
  }

  const hashString = (str: string): number => {
    let hash = 0
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i)
      hash = ((hash << 5) - hash) + char
      hash = hash & hash // Convert to 32-bit integer
    }
    return Math.abs(hash)
  }

  const getStoredVariants = (): Record<string, string> => {
    try {
      const stored = localStorage.getItem('ab-test-variants')
      return stored ? JSON.parse(stored) : {}
    } catch {
      return {}
    }
  }

  const storeVariants = (variants: Record<string, string>) => {
    try {
      localStorage.setItem('ab-test-variants', JSON.stringify(variants))
    } catch (error) {
      console.error('Failed to store A/B test variants:', error)
    }
  }

  // Get the variant configuration for a test
  const getVariant = useCallback((testId: string): ABVariant | null => {
    const test = state.activeTests[testId]
    const variantId = state.userVariants[testId]
    
    if (!test || !variantId) return null
    
    return test.variants.find(v => v.id === variantId) || null
  }, [state.activeTests, state.userVariants])

  // Get variant config value with fallback
  const getVariantValue = useCallback(<T>(
    testId: string,
    configKey: string,
    fallback: T
  ): T => {
    const variant = getVariant(testId)
    return variant?.config[configKey] ?? fallback
  }, [getVariant])

  // Track conversion for a test
  const trackConversion = useCallback(async (
    testId: string,
    metric: string,
    value: number = 1
  ) => {
    const variant = getVariant(testId)
    if (!variant) return

    const conversionKey = `${testId}_${metric}`
    
    setState(prev => ({
      ...prev,
      conversions: {
        ...prev.conversions,
        [conversionKey]: (prev.conversions[conversionKey] || 0) + value,
      },
    }))

    // Track analytics event
    track('ab_test_conversion', {
      testId,
      variantId: variant.id,
      metric,
      value,
      testName: state.activeTests[testId]?.name,
      variantName: variant.name,
    })

    // Send to backend
    try {
      await fetch('/api/ab-testing/conversions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          testId,
          variantId: variant.id,
          metric,
          value,
          timestamp: Date.now(),
        }),
      })
    } catch (error) {
      console.error('Failed to track A/B test conversion:', error)
    }
  }, [getVariant, track, state.activeTests])

  // Check if user is in a specific test
  const isInTest = useCallback((testId: string): boolean => {
    return testId in state.userVariants
  }, [state.userVariants])

  // Check if user is in a specific variant
  const isInVariant = useCallback((testId: string, variantId: string): boolean => {
    return state.userVariants[testId] === variantId
  }, [state.userVariants])

  // Get all active test configurations for debugging
  const getTestConfigurations = useCallback(() => {
    return Object.entries(state.userVariants).map(([testId, variantId]) => {
      const test = state.activeTests[testId]
      const variant = test?.variants.find(v => v.id === variantId)
      
      return {
        testId,
        testName: test?.name,
        variantId,
        variantName: variant?.name,
        config: variant?.config,
        isControl: variant?.isControl,
      }
    })
  }, [state.activeTests, state.userVariants])

  return {
    // State
    isLoading,
    activeTests: state.activeTests,
    userVariants: state.userVariants,
    conversions: state.conversions,

    // Variant methods
    getVariant,
    getVariantValue,
    isInTest,
    isInVariant,

    // Conversion tracking
    trackConversion,

    // Utility methods
    getTestConfigurations,
    refreshTests: loadActiveTests,
  }
}

// React component wrapper for A/B testing
export function ABTestVariant({
  testId,
  variantId,
  children,
  fallback = null,
}: {
  testId: string
  variantId: string
  children: React.ReactNode
  fallback?: React.ReactNode
}) {
  const { isInVariant } = useABTesting()
  
  if (isInVariant(testId, variantId)) {
    return <>{children}</>
  }
  
  return <>{fallback}</>
}

// Hook for conditional rendering based on A/B test
export function useABTestVariant(testId: string, variantId: string): boolean {
  const { isInVariant } = useABTesting()
  return isInVariant(testId, variantId)
}

// Hook for getting feature flags from A/B tests
export function useFeatureFlag(testId: string, flagName: string, defaultValue: boolean = false): boolean {
  const { getVariantValue } = useABTesting()
  return getVariantValue(testId, flagName, defaultValue)
}