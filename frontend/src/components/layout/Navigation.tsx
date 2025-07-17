'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Bars3Icon,
  XMarkIcon,
  HomeIcon,
  MusicalNoteIcon,
  SpeakerWaveIcon,
  UserGroupIcon,
  UserCircleIcon,
  Cog6ToothIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline'
import { Button } from '@/components/ui/Button'
import { cn } from '@/utils/cn'

interface NavigationItem {
  name: string
  href: string
  icon: React.ComponentType<{ className?: string }>
  badge?: string
}

const mainNavItems: NavigationItem[] = [
  { name: 'Home', href: '/', icon: HomeIcon },
  { name: 'Studio', href: '/studio', icon: MusicalNoteIcon },
  { name: 'Editor', href: '/studio/editor', icon: SpeakerWaveIcon },
  { name: 'Community', href: '/community', icon: UserGroupIcon },
  { name: 'Discover', href: '/discover', icon: MagnifyingGlassIcon },
]

const userNavItems: NavigationItem[] = [
  { name: 'Profile', href: '/profile', icon: UserCircleIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
]

export function Navigation() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const pathname = usePathname()

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false)
  }

  const isActiveLink = (href: string) => {
    if (href === '/') {
      return pathname === href
    }
    return pathname.startsWith(href)
  }

  return (
    <>
      {/* Desktop Navigation */}
      <nav className="hidden lg:flex lg:flex-col lg:w-64 lg:fixed lg:inset-y-0 lg:bg-gray-900 lg:border-r lg:border-gray-700">
        <div className="flex items-center h-16 px-6 border-b border-gray-700">
          <Link href="/" className="flex items-center space-x-2">
            <MusicalNoteIcon className="h-8 w-8 text-purple-500" />
            <span className="text-xl font-bold text-white">MusicGen AI</span>
          </Link>
        </div>

        <div className="flex-1 flex flex-col overflow-y-auto pt-5 pb-4">
          <div className="flex-1 px-3 space-y-1">
            {mainNavItems.map((item) => {
              const Icon = item.icon
              const isActive = isActiveLink(item.href)
              
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors',
                    isActive
                      ? 'bg-purple-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  )}
                >
                  <Icon
                    className={cn(
                      'mr-3 flex-shrink-0 h-6 w-6',
                      isActive ? 'text-white' : 'text-gray-400 group-hover:text-gray-300'
                    )}
                  />
                  {item.name}
                  {item.badge && (
                    <span className="ml-auto inline-block py-0.5 px-3 text-xs font-medium rounded-full bg-purple-100 text-purple-800">
                      {item.badge}
                    </span>
                  )}
                </Link>
              )
            })}
          </div>

          <div className="mt-6 pt-6 border-t border-gray-700">
            <div className="px-3 space-y-1">
              {userNavItems.map((item) => {
                const Icon = item.icon
                const isActive = isActiveLink(item.href)
                
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={cn(
                      'group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors',
                      isActive
                        ? 'bg-purple-600 text-white'
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                    )}
                  >
                    <Icon
                      className={cn(
                        'mr-3 flex-shrink-0 h-6 w-6',
                        isActive ? 'text-white' : 'text-gray-400 group-hover:text-gray-300'
                      )}
                    />
                    {item.name}
                  </Link>
                )
              })}
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile Navigation */}
      <div className="lg:hidden">
        {/* Mobile Header */}
        <div className="flex items-center justify-between h-16 px-4 bg-gray-900 border-b border-gray-700">
          <Link href="/" className="flex items-center space-x-2">
            <MusicalNoteIcon className="h-8 w-8 text-purple-500" />
            <span className="text-xl font-bold text-white">MusicGen AI</span>
          </Link>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleMobileMenu}
            className="text-gray-400 hover:text-white"
          >
            {isMobileMenuOpen ? (
              <XMarkIcon className="h-6 w-6" />
            ) : (
              <Bars3Icon className="h-6 w-6" />
            )}
          </Button>
        </div>

        {/* Mobile Menu Overlay */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <>
              {/* Backdrop */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={closeMobileMenu}
                className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
              />

              {/* Mobile Menu */}
              <motion.div
                initial={{ x: '-100%' }}
                animate={{ x: 0 }}
                exit={{ x: '-100%' }}
                transition={{ type: 'tween', duration: 0.3 }}
                className="fixed inset-y-0 left-0 z-50 w-80 bg-gray-900 overflow-y-auto lg:hidden"
              >
                <div className="flex items-center h-16 px-6 border-b border-gray-700">
                  <Link href="/" onClick={closeMobileMenu} className="flex items-center space-x-2">
                    <MusicalNoteIcon className="h-8 w-8 text-purple-500" />
                    <span className="text-xl font-bold text-white">MusicGen AI</span>
                  </Link>
                </div>

                <div className="pt-5 pb-4">
                  <div className="px-3 space-y-1">
                    {mainNavItems.map((item) => {
                      const Icon = item.icon
                      const isActive = isActiveLink(item.href)
                      
                      return (
                        <Link
                          key={item.name}
                          href={item.href}
                          onClick={closeMobileMenu}
                          className={cn(
                            'group flex items-center px-2 py-2 text-base font-medium rounded-md transition-colors',
                            isActive
                              ? 'bg-purple-600 text-white'
                              : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                          )}
                        >
                          <Icon
                            className={cn(
                              'mr-4 flex-shrink-0 h-6 w-6',
                              isActive ? 'text-white' : 'text-gray-400 group-hover:text-gray-300'
                            )}
                          />
                          {item.name}
                          {item.badge && (
                            <span className="ml-auto inline-block py-0.5 px-3 text-xs font-medium rounded-full bg-purple-100 text-purple-800">
                              {item.badge}
                            </span>
                          )}
                        </Link>
                      )
                    })}
                  </div>

                  <div className="mt-6 pt-6 border-t border-gray-700">
                    <div className="px-3 space-y-1">
                      {userNavItems.map((item) => {
                        const Icon = item.icon
                        const isActive = isActiveLink(item.href)
                        
                        return (
                          <Link
                            key={item.name}
                            href={item.href}
                            onClick={closeMobileMenu}
                            className={cn(
                              'group flex items-center px-2 py-2 text-base font-medium rounded-md transition-colors',
                              isActive
                                ? 'bg-purple-600 text-white'
                                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                            )}
                          >
                            <Icon
                              className={cn(
                                'mr-4 flex-shrink-0 h-6 w-6',
                                isActive ? 'text-white' : 'text-gray-400 group-hover:text-gray-300'
                              )}
                            />
                            {item.name}
                          </Link>
                        )
                      })}
                    </div>
                  </div>
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Navigation for Mobile (Alternative/Additional) */}
      <div className="lg:hidden fixed bottom-0 left-0 right-0 z-30 bg-gray-900 border-t border-gray-700">
        <div className="grid grid-cols-5 py-2">
          {mainNavItems.slice(0, 5).map((item) => {
            const Icon = item.icon
            const isActive = isActiveLink(item.href)
            
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  'flex flex-col items-center justify-center py-2 px-1 text-xs transition-colors',
                  isActive ? 'text-purple-500' : 'text-gray-400 hover:text-gray-300'
                )}
              >
                <Icon className="h-5 w-5 mb-1" />
                <span className="truncate">{item.name}</span>
              </Link>
            )
          })}
        </div>
      </div>
    </>
  )
}