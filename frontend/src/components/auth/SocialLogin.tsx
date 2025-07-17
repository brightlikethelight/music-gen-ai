'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/Button'

interface SocialProvider {
  id: string
  name: string
  icon: string
  color: string
  hoverColor: string
}

const socialProviders: SocialProvider[] = [
  {
    id: 'google',
    name: 'Google',
    icon: '🔍',
    color: 'bg-white text-gray-900 border border-gray-300',
    hoverColor: 'hover:bg-gray-50',
  },
  {
    id: 'apple',
    name: 'Apple',
    icon: '🍎',
    color: 'bg-black text-white border border-gray-600',
    hoverColor: 'hover:bg-gray-900',
  },
  {
    id: 'spotify',
    name: 'Spotify',
    icon: '🎵',
    color: 'bg-green-600 text-white border border-green-600',
    hoverColor: 'hover:bg-green-700',
  },
]

export function SocialLogin() {
  const [loadingProvider, setLoadingProvider] = useState<string | null>(null)

  const handleSocialLogin = async (providerId: string) => {
    setLoadingProvider(providerId)

    try {
      // In a real implementation, this would redirect to the OAuth provider
      // For now, we'll simulate the process
      
      if (providerId === 'google') {
        window.location.href = '/api/auth/google'
      } else if (providerId === 'apple') {
        window.location.href = '/api/auth/apple'
      } else if (providerId === 'spotify') {
        window.location.href = '/api/auth/spotify'
      }
    } catch (error) {
      console.error(`${providerId} login failed:`, error)
      setLoadingProvider(null)
    }
  }

  return (
    <div className="space-y-3">
      {socialProviders.map((provider) => (
        <motion.div
          key={provider.id}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Button
            type="button"
            variant="outline"
            onClick={() => handleSocialLogin(provider.id)}
            disabled={loadingProvider !== null}
            className={`w-full ${provider.color} ${provider.hoverColor} transition-colors duration-200 relative`}
          >
            {loadingProvider === provider.id ? (
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                <span>Connecting...</span>
              </div>
            ) : (
              <div className="flex items-center justify-center space-x-3">
                <span className="text-lg">{provider.icon}</span>
                <span className="font-medium">Continue with {provider.name}</span>
              </div>
            )}
          </Button>
        </motion.div>
      ))}
    </div>
  )
}

// Google OAuth component for more detailed implementation
export function GoogleOAuthButton() {
  const [isLoading, setIsLoading] = useState(false)

  const handleGoogleLogin = async () => {
    setIsLoading(true)

    try {
      // Initialize Google OAuth
      if (typeof window !== 'undefined' && window.google) {
        window.google.accounts.id.initialize({
          client_id: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
          callback: handleGoogleResponse,
        })

        window.google.accounts.id.prompt()
      } else {
        // Fallback to redirect method
        window.location.href = `/api/auth/google?redirect_uri=${encodeURIComponent(window.location.origin + '/auth/callback')}`
      }
    } catch (error) {
      console.error('Google login failed:', error)
      setIsLoading(false)
    }
  }

  const handleGoogleResponse = async (response: any) => {
    try {
      // Send the credential to your backend
      const result = await fetch('/api/auth/google/callback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          credential: response.credential,
        }),
      })

      if (result.ok) {
        const data = await result.json()
        // Handle successful authentication
        window.location.href = '/studio'
      } else {
        throw new Error('Authentication failed')
      }
    } catch (error) {
      console.error('Google authentication error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Button
      type="button"
      variant="outline"
      onClick={handleGoogleLogin}
      disabled={isLoading}
      className="w-full bg-white text-gray-900 border border-gray-300 hover:bg-gray-50"
    >
      {isLoading ? (
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
          <span>Connecting...</span>
        </div>
      ) : (
        <div className="flex items-center justify-center space-x-3">
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path
              fill="currentColor"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="currentColor"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="currentColor"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
            />
            <path
              fill="currentColor"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
            />
          </svg>
          <span className="font-medium">Continue with Google</span>
        </div>
      )}
    </Button>
  )
}