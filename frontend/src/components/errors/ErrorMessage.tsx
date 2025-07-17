'use client'

import { motion } from 'framer-motion'
import { 
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'

export type ErrorSeverity = 'error' | 'warning' | 'info' | 'success'

interface ErrorMessageProps {
  message: string
  severity?: ErrorSeverity
  title?: string
  onDismiss?: () => void
  dismissible?: boolean
  className?: string
  children?: React.ReactNode
}

const severityConfig = {
  error: {
    icon: XCircleIcon,
    bgColor: 'bg-red-900/20',
    borderColor: 'border-red-700',
    iconColor: 'text-red-400',
    textColor: 'text-red-200',
    titleColor: 'text-red-300',
  },
  warning: {
    icon: ExclamationTriangleIcon,
    bgColor: 'bg-yellow-900/20',
    borderColor: 'border-yellow-700',
    iconColor: 'text-yellow-400',
    textColor: 'text-yellow-200',
    titleColor: 'text-yellow-300',
  },
  info: {
    icon: InformationCircleIcon,
    bgColor: 'bg-blue-900/20',
    borderColor: 'border-blue-700',
    iconColor: 'text-blue-400',
    textColor: 'text-blue-200',
    titleColor: 'text-blue-300',
  },
  success: {
    icon: CheckCircleIcon,
    bgColor: 'bg-green-900/20',
    borderColor: 'border-green-700',
    iconColor: 'text-green-400',
    textColor: 'text-green-200',
    titleColor: 'text-green-300',
  },
}

export function ErrorMessage({
  message,
  severity = 'error',
  title,
  onDismiss,
  dismissible = true,
  className = '',
  children,
}: ErrorMessageProps) {
  const config = severityConfig[severity]
  const Icon = config.icon

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={`${config.bgColor} ${config.borderColor} border rounded-lg p-4 ${className}`}
      role="alert"
      aria-live="polite"
    >
      <div className="flex">
        <div className="flex-shrink-0">
          <Icon className={`h-5 w-5 ${config.iconColor}`} aria-hidden="true" />
        </div>
        
        <div className="ml-3 flex-1">
          {title && (
            <h3 className={`text-sm font-medium ${config.titleColor} mb-1`}>
              {title}
            </h3>
          )}
          
          <div className={`text-sm ${config.textColor}`}>
            <p>{message}</p>
            {children && (
              <div className="mt-2">
                {children}
              </div>
            )}
          </div>
        </div>

        {dismissible && onDismiss && (
          <div className="ml-auto pl-3">
            <div className="-mx-1.5 -my-1.5">
              <button
                type="button"
                onClick={onDismiss}
                className={`inline-flex rounded-md p-1.5 ${config.iconColor} hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-gray-600`}
                aria-label="Dismiss"
              >
                <XMarkIcon className="h-5 w-5" aria-hidden="true" />
              </button>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  )
}