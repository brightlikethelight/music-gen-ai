'use client'

import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  WifiIcon,
  ArrowPathIcon,
  SignalSlashIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'

interface NetworkErrorBoundaryProps {
  children: React.ReactNode
  onRetry?: () => void
  fallback?: React.ReactNode
}

export function NetworkErrorBoundary({ 
  children, 
  onRetry,
  fallback 
}: NetworkErrorBoundaryProps) {
  const [isOnline, setIsOnline] = useState(true)
  const [hasNetworkError, setHasNetworkError] = useState(false)
  const [retryCount, setRetryCount] = useState(0)
  const [isRetrying, setIsRetrying] = useState(false)

  useEffect(() => {
    // Check initial online status
    setIsOnline(navigator.onLine)

    // Handle online/offline events
    const handleOnline = () => {
      setIsOnline(true)
      setHasNetworkError(false)
      setRetryCount(0)
    }

    const handleOffline = () => {
      setIsOnline(false)
      setHasNetworkError(true)
    }

    // Handle fetch errors
    const handleFetchError = (event: PromiseRejectionEvent) => {
      if (event.reason instanceof TypeError && event.reason.message.includes('fetch')) {
        setHasNetworkError(true)
        event.preventDefault() // Prevent unhandled rejection warning
      }
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    window.addEventListener('unhandledrejection', handleFetchError)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
      window.removeEventListener('unhandledrejection', handleFetchError)
    }
  }, [])

  const handleRetry = async () => {
    setIsRetrying(true)
    setRetryCount(prev => prev + 1)

    // Simulate network check
    await new Promise(resolve => setTimeout(resolve, 1000))

    if (navigator.onLine) {
      setHasNetworkError(false)
      setIsOnline(true)
      if (onRetry) {
        onRetry()
      }
    }

    setIsRetrying(false)
  }

  const getRetryDelay = () => {
    // Exponential backoff: 5s, 10s, 20s, 40s...
    return Math.min(5 * Math.pow(2, retryCount), 60)
  }

  if (!isOnline || hasNetworkError) {
    if (fallback) {
      return <>{fallback}</>
    }

    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className="min-h-screen bg-gray-900 flex items-center justify-center p-4"
        >
          <div className="max-w-md w-full text-center">
            {/* Icon */}
            <div className="mb-8">
              {!isOnline ? (
                <div className="relative inline-block">
                  <WifiIcon className="h-24 w-24 text-gray-600" />
                  <SignalSlashIcon className="h-12 w-12 text-red-400 absolute bottom-0 right-0" />
                </div>
              ) : (
                <ExclamationTriangleIcon className="h-24 w-24 text-yellow-400 mx-auto" />
              )}
            </div>

            {/* Title */}
            <h2 className="text-2xl font-bold text-white mb-2">
              {!isOnline ? 'No Internet Connection' : 'Network Error'}
            </h2>

            {/* Description */}
            <p className="text-gray-400 mb-8">
              {!isOnline 
                ? 'Please check your internet connection and try again.'
                : 'We\'re having trouble connecting to our servers. Please try again.'}
            </p>

            {/* Retry Information */}
            {retryCount > 0 && (
              <p className="text-sm text-gray-500 mb-4">
                Retry attempt {retryCount} • Next retry in {getRetryDelay()} seconds
              </p>
            )}

            {/* Actions */}
            <div className="space-y-4">
              <Button
                onClick={handleRetry}
                disabled={isRetrying}
                className="bg-purple-600 hover:bg-purple-700 w-full"
              >
                {isRetrying ? (
                  <>
                    <ArrowPathIcon className="h-4 w-4 mr-2 animate-spin" />
                    Checking Connection...
                  </>
                ) : (
                  <>
                    <ArrowPathIcon className="h-4 w-4 mr-2" />
                    Try Again
                  </>
                )}
              </Button>

              <Button
                onClick={() => window.location.reload()}
                variant="outline"
                className="border-gray-600 text-gray-300 hover:bg-gray-700 w-full"
              >
                Refresh Page
              </Button>
            </div>

            {/* Troubleshooting Tips */}
            <div className="mt-8 text-left bg-gray-800 rounded-lg p-4 border border-gray-700">
              <h3 className="text-sm font-medium text-gray-300 mb-2">Troubleshooting Tips:</h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Check your Wi-Fi or mobile data connection</li>
                <li>• Disable VPN if you're using one</li>
                <li>• Try opening other websites to test connectivity</li>
                <li>• Clear your browser cache and cookies</li>
              </ul>
            </div>

            {/* Status Indicator */}
            <div className="mt-6 flex items-center justify-center space-x-2 text-sm">
              <div className={`h-2 w-2 rounded-full ${isOnline ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-gray-400">
                {isOnline ? 'Browser Online' : 'Browser Offline'}
              </span>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    )
  }

  return <>{children}</>
}