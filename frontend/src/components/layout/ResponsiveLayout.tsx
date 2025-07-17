'use client'

import { ReactNode } from 'react'
import { Navigation } from './Navigation'
import { cn } from '@/utils/cn'

interface ResponsiveLayoutProps {
  children: ReactNode
  className?: string
  fullHeight?: boolean
  noPadding?: boolean
}

export function ResponsiveLayout({ 
  children, 
  className,
  fullHeight = false,
  noPadding = false
}: ResponsiveLayoutProps) {
  return (
    <div className="min-h-screen bg-gray-900">
      <Navigation />
      
      {/* Main Content */}
      <div className={cn(
        'lg:pl-64', // Account for desktop sidebar
        fullHeight ? 'h-screen' : 'min-h-screen',
        className
      )}>
        <div className={cn(
          'lg:pt-0 pt-16 pb-20 lg:pb-0', // Account for mobile header and bottom nav
          !noPadding && 'px-4 sm:px-6 lg:px-8 py-6',
          fullHeight && !noPadding && 'h-full',
          noPadding && fullHeight && 'h-screen'
        )}>
          {children}
        </div>
      </div>
    </div>
  )
}

// Utility component for responsive containers
export function ResponsiveContainer({ 
  children, 
  className,
  size = 'default'
}: {
  children: ReactNode
  className?: string
  size?: 'sm' | 'default' | 'lg' | 'xl' | 'full'
}) {
  const sizeClasses = {
    sm: 'max-w-2xl',
    default: 'max-w-7xl',
    lg: 'max-w-8xl', 
    xl: 'max-w-screen-2xl',
    full: 'max-w-none'
  }

  return (
    <div className={cn(
      'mx-auto w-full',
      sizeClasses[size],
      className
    )}>
      {children}
    </div>
  )
}

// Utility component for responsive grids
export function ResponsiveGrid({ 
  children, 
  className,
  cols = { sm: 1, md: 2, lg: 3, xl: 4 }
}: {
  children: ReactNode
  className?: string
  cols?: {
    sm?: number
    md?: number
    lg?: number
    xl?: number
  }
}) {
  const gridClasses = [
    'grid gap-6',
    `grid-cols-${cols.sm || 1}`,
    cols.md && `md:grid-cols-${cols.md}`,
    cols.lg && `lg:grid-cols-${cols.lg}`,
    cols.xl && `xl:grid-cols-${cols.xl}`,
  ].filter(Boolean).join(' ')

  return (
    <div className={cn(gridClasses, className)}>
      {children}
    </div>
  )
}

// Utility component for responsive sections
export function ResponsiveSection({ 
  children, 
  className,
  spacing = 'default'
}: {
  children: ReactNode
  className?: string
  spacing?: 'none' | 'sm' | 'default' | 'lg' | 'xl'
}) {
  const spacingClasses = {
    none: '',
    sm: 'py-8',
    default: 'py-12',
    lg: 'py-16',
    xl: 'py-24'
  }

  return (
    <section className={cn(
      spacingClasses[spacing],
      className
    )}>
      {children}
    </section>
  )
}

// Hook for responsive breakpoints
import { useState, useEffect } from 'react'

export function useResponsive() {
  const [breakpoint, setBreakpoint] = useState<'sm' | 'md' | 'lg' | 'xl' | '2xl'>('md')
  const [isMobile, setIsMobile] = useState(false)
  const [isTablet, setIsTablet] = useState(false)
  const [isDesktop, setIsDesktop] = useState(false)

  useEffect(() => {
    const updateBreakpoint = () => {
      const width = window.innerWidth
      
      if (width < 640) {
        setBreakpoint('sm')
        setIsMobile(true)
        setIsTablet(false)
        setIsDesktop(false)
      } else if (width < 768) {
        setBreakpoint('md')
        setIsMobile(true)
        setIsTablet(false)
        setIsDesktop(false)
      } else if (width < 1024) {
        setBreakpoint('lg')
        setIsMobile(false)
        setIsTablet(true)
        setIsDesktop(false)
      } else if (width < 1280) {
        setBreakpoint('xl')
        setIsMobile(false)
        setIsTablet(false)
        setIsDesktop(true)
      } else {
        setBreakpoint('2xl')
        setIsMobile(false)
        setIsTablet(false)
        setIsDesktop(true)
      }
    }

    updateBreakpoint()
    window.addEventListener('resize', updateBreakpoint)
    
    return () => window.removeEventListener('resize', updateBreakpoint)
  }, [])

  return {
    breakpoint,
    isMobile,
    isTablet,
    isDesktop,
    isSmallScreen: breakpoint === 'sm' || breakpoint === 'md',
    isLargeScreen: breakpoint === 'xl' || breakpoint === '2xl'
  }
}