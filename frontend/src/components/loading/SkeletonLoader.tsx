'use client'

import { motion } from 'framer-motion'

interface SkeletonLoaderProps {
  variant?: 'text' | 'circular' | 'rectangular' | 'card' | 'list'
  width?: string | number
  height?: string | number
  count?: number
  className?: string
  animate?: boolean
}

const shimmer = {
  initial: { backgroundPosition: '-200% 0' },
  animate: {
    backgroundPosition: '200% 0',
    transition: {
      duration: 1.5,
      repeat: Infinity,
      ease: 'linear',
    },
  },
}

export function SkeletonLoader({
  variant = 'text',
  width,
  height,
  count = 1,
  className = '',
  animate = true,
}: SkeletonLoaderProps) {
  const baseClasses = `bg-gray-700 ${animate ? 'bg-gradient-to-r from-gray-700 via-gray-600 to-gray-700 bg-[length:200%_100%]' : ''}`

  const renderSkeleton = () => {
    switch (variant) {
      case 'text':
        return (
          <motion.div
            className={`h-4 rounded ${baseClasses} ${className}`}
            style={{ width: width || '100%' }}
            variants={animate ? shimmer : undefined}
            initial="initial"
            animate="animate"
          />
        )

      case 'circular':
        return (
          <motion.div
            className={`rounded-full ${baseClasses} ${className}`}
            style={{
              width: width || 40,
              height: height || width || 40,
            }}
            variants={animate ? shimmer : undefined}
            initial="initial"
            animate="animate"
          />
        )

      case 'rectangular':
        return (
          <motion.div
            className={`rounded ${baseClasses} ${className}`}
            style={{
              width: width || '100%',
              height: height || 60,
            }}
            variants={animate ? shimmer : undefined}
            initial="initial"
            animate="animate"
          />
        )

      case 'card':
        return (
          <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
            <motion.div
              className={`h-32 rounded mb-4 ${baseClasses}`}
              variants={animate ? shimmer : undefined}
              initial="initial"
              animate="animate"
            />
            <motion.div
              className={`h-4 rounded mb-2 ${baseClasses}`}
              style={{ width: '70%' }}
              variants={animate ? shimmer : undefined}
              initial="initial"
              animate="animate"
            />
            <motion.div
              className={`h-4 rounded ${baseClasses}`}
              style={{ width: '50%' }}
              variants={animate ? shimmer : undefined}
              initial="initial"
              animate="animate"
            />
          </div>
        )

      case 'list':
        return (
          <div className={`space-y-3 ${className}`}>
            {Array.from({ length: count }).map((_, index) => (
              <div key={index} className="flex items-center space-x-3">
                <motion.div
                  className={`h-10 w-10 rounded-full flex-shrink-0 ${baseClasses}`}
                  variants={animate ? shimmer : undefined}
                  initial="initial"
                  animate="animate"
                />
                <div className="flex-1 space-y-2">
                  <motion.div
                    className={`h-4 rounded ${baseClasses}`}
                    style={{ width: '60%' }}
                    variants={animate ? shimmer : undefined}
                    initial="initial"
                    animate="animate"
                  />
                  <motion.div
                    className={`h-3 rounded ${baseClasses}`}
                    style={{ width: '40%' }}
                    variants={animate ? shimmer : undefined}
                    initial="initial"
                    animate="animate"
                  />
                </div>
              </div>
            ))}
          </div>
        )

      default:
        return null
    }
  }

  if (count > 1 && variant !== 'list') {
    return (
      <div className="space-y-2">
        {Array.from({ length: count }).map((_, index) => (
          <div key={index}>{renderSkeleton()}</div>
        ))}
      </div>
    )
  }

  return renderSkeleton()
}