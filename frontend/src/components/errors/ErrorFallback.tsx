'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ExclamationTriangleIcon, 
  ArrowPathIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ClipboardDocumentIcon,
  CheckIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'

interface ErrorFallbackProps {
  error: Error
  errorInfo?: React.ErrorInfo | null
  reset: () => void
  minimal?: boolean
}

export function ErrorFallback({ error, errorInfo, reset, minimal = false }: ErrorFallbackProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [copied, setCopied] = useState(false)

  const copyErrorToClipboard = async () => {
    const errorText = `
Error: ${error.message}

Stack Trace:
${error.stack}

Component Stack:
${errorInfo?.componentStack || 'Not available'}
    `.trim()

    try {
      await navigator.clipboard.writeText(errorText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy error:', err)
    }
  }

  const handleReset = () => {
    // Clear any error state before resetting
    setShowDetails(false)
    setCopied(false)
    reset()
  }

  if (minimal) {
    return (
      <div className="flex items-center justify-center p-4">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-400 mx-auto mb-4" />
          <p className="text-gray-400 mb-4">Something went wrong</p>
          <Button onClick={handleReset} size="sm" variant="outline">
            Try Again
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="max-w-2xl w-full"
      >
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="bg-red-900/20 p-4 rounded-full">
                <ExclamationTriangleIcon className="h-16 w-16 text-red-400" />
              </div>
            </div>
            <CardTitle className="text-2xl font-bold text-white mb-2">
              Oops! Something went wrong
            </CardTitle>
            <p className="text-gray-400">
              We encountered an unexpected error. Don't worry, your work has been saved.
            </p>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Error Message */}
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
              <h3 className="text-sm font-medium text-gray-300 mb-2">Error Message</h3>
              <p className="text-red-400 font-mono text-sm break-words">
                {error.message || 'Unknown error occurred'}
              </p>
            </div>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-3">
              <Button
                onClick={handleReset}
                className="flex-1 bg-purple-600 hover:bg-purple-700"
              >
                <ArrowPathIcon className="h-4 w-4 mr-2" />
                Try Again
              </Button>
              
              <Button
                onClick={() => window.location.href = '/'}
                variant="outline"
                className="flex-1 border-gray-600 text-gray-300 hover:bg-gray-700"
              >
                Go to Home
              </Button>
            </div>

            {/* Technical Details */}
            <div className="border-t border-gray-700 pt-4">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="flex items-center justify-between w-full text-left"
              >
                <span className="text-sm text-gray-400">Technical Details</span>
                {showDetails ? (
                  <ChevronUpIcon className="h-4 w-4 text-gray-400" />
                ) : (
                  <ChevronDownIcon className="h-4 w-4 text-gray-400" />
                )}
              </button>

              <AnimatePresence>
                {showDetails && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4 space-y-4"
                  >
                    {/* Stack Trace */}
                    <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-sm font-medium text-gray-300">Stack Trace</h4>
                        <button
                          onClick={copyErrorToClipboard}
                          className="text-gray-400 hover:text-white transition-colors"
                          title="Copy error details"
                        >
                          {copied ? (
                            <CheckIcon className="h-4 w-4 text-green-400" />
                          ) : (
                            <ClipboardDocumentIcon className="h-4 w-4" />
                          )}
                        </button>
                      </div>
                      <pre className="text-xs text-gray-500 overflow-auto max-h-40">
                        {error.stack || 'No stack trace available'}
                      </pre>
                    </div>

                    {/* Component Stack */}
                    {errorInfo?.componentStack && (
                      <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
                        <h4 className="text-sm font-medium text-gray-300 mb-2">Component Stack</h4>
                        <pre className="text-xs text-gray-500 overflow-auto max-h-40">
                          {errorInfo.componentStack}
                        </pre>
                      </div>
                    )}

                    {/* Additional Info */}
                    <div className="text-xs text-gray-500 space-y-1">
                      <p>Time: {new Date().toLocaleString()}</p>
                      <p>URL: {typeof window !== 'undefined' ? window.location.href : 'N/A'}</p>
                      <p>User Agent: {typeof navigator !== 'undefined' ? navigator.userAgent : 'N/A'}</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Help Text */}
            <div className="text-center text-sm text-gray-400">
              <p>If this problem persists, please contact our support team.</p>
              <a 
                href="mailto:support@musicgen.ai" 
                className="text-purple-400 hover:text-purple-300 underline"
              >
                support@musicgen.ai
              </a>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}