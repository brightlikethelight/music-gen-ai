'use client'

import { ReactNode, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ResponsiveContainerProps {
  children: ReactNode
  breakpoint?: 'sm' | 'md' | 'lg' | 'xl' | '2xl'
  mobileComponent?: ReactNode
  desktopComponent?: ReactNode
  className?: string
}

const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
}

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined') return

    const media = window.matchMedia(query)
    setMatches(media.matches)

    const listener = (event: MediaQueryListEvent) => {
      setMatches(event.matches)
    }

    media.addEventListener('change', listener)
    return () => media.removeEventListener('change', listener)
  }, [query])

  return matches
}

export function useBreakpoint(breakpoint: keyof typeof breakpoints = 'md'): boolean {
  return useMediaQuery(`(min-width: ${breakpoints[breakpoint]}px)`)
}

export function useIsMobile(): boolean {
  return !useBreakpoint('md')
}

export function ResponsiveContainer({
  children,
  breakpoint = 'md',
  mobileComponent,
  desktopComponent,
  className = '',
}: ResponsiveContainerProps) {
  const isDesktop = useBreakpoint(breakpoint)

  if (mobileComponent && desktopComponent) {
    return (
      <AnimatePresence mode="wait">
        {isDesktop ? (
          <motion.div
            key="desktop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={className}
          >
            {desktopComponent}
          </motion.div>
        ) : (
          <motion.div
            key="mobile"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={className}
          >
            {mobileComponent}
          </motion.div>
        )}
      </AnimatePresence>
    )
  }

  return <div className={className}>{children}</div>
}

// Mobile-specific touch interactions component
interface TouchInteractionProps {
  onSwipeLeft?: () => void
  onSwipeRight?: () => void
  onSwipeUp?: () => void
  onSwipeDown?: () => void
  onTap?: () => void
  onDoubleTap?: () => void
  onLongPress?: () => void
  threshold?: number
  children: ReactNode
  className?: string
}

export function TouchInteraction({
  onSwipeLeft,
  onSwipeRight,
  onSwipeUp,
  onSwipeDown,
  onTap,
  onDoubleTap,
  onLongPress,
  threshold = 50,
  children,
  className = '',
}: TouchInteractionProps) {
  const [touchStart, setTouchStart] = useState<{ x: number; y: number } | null>(null)
  const [lastTap, setLastTap] = useState(0)
  const [longPressTimer, setLongPressTimer] = useState<NodeJS.Timeout | null>(null)

  const handleTouchStart = (e: React.TouchEvent) => {
    const touch = e.touches[0]
    setTouchStart({ x: touch.clientX, y: touch.clientY })

    // Long press detection
    if (onLongPress) {
      const timer = setTimeout(() => {
        onLongPress()
      }, 500)
      setLongPressTimer(timer)
    }
  }

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (longPressTimer) {
      clearTimeout(longPressTimer)
      setLongPressTimer(null)
    }

    if (!touchStart) return

    const touch = e.changedTouches[0]
    const deltaX = touch.clientX - touchStart.x
    const deltaY = touch.clientY - touchStart.y

    // Swipe detection
    if (Math.abs(deltaX) > threshold || Math.abs(deltaY) > threshold) {
      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        // Horizontal swipe
        if (deltaX > threshold && onSwipeRight) {
          onSwipeRight()
        } else if (deltaX < -threshold && onSwipeLeft) {
          onSwipeLeft()
        }
      } else {
        // Vertical swipe
        if (deltaY > threshold && onSwipeDown) {
          onSwipeDown()
        } else if (deltaY < -threshold && onSwipeUp) {
          onSwipeUp()
        }
      }
    } else {
      // Tap detection
      const currentTime = Date.now()
      const tapDelta = currentTime - lastTap

      if (tapDelta < 300 && onDoubleTap) {
        // Double tap
        onDoubleTap()
        setLastTap(0)
      } else {
        // Single tap
        if (onTap) {
          onTap()
        }
        setLastTap(currentTime)
      }
    }

    setTouchStart(null)
  }

  const handleTouchCancel = () => {
    if (longPressTimer) {
      clearTimeout(longPressTimer)
      setLongPressTimer(null)
    }
    setTouchStart(null)
  }

  return (
    <div
      className={className}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      onTouchCancel={handleTouchCancel}
    >
      {children}
    </div>
  )
}