'use client'

import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useAnalytics } from '@/components/analytics/AnalyticsProvider'

// Types
interface User {
  id: string
  email: string
  username: string
  firstName?: string
  lastName?: string
  tier: 'free' | 'pro' | 'enterprise'
  isVerified: boolean
  createdAt: string
  lastLogin: string
}

interface AuthContextType {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  csrfToken: string | null
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>
  register: (userData: RegisterData) => Promise<{ success: boolean; error?: string }>
  logout: () => Promise<void>
  refreshSession: () => Promise<void>
  updateUser: (updates: Partial<User>) => Promise<void>
  deleteAccount: () => Promise<void>
  resendVerification: () => Promise<void>
  requestPasswordReset: (email: string) => Promise<{ success: boolean; error?: string }>
  resetPassword: (token: string, password: string) => Promise<{ success: boolean; error?: string }>
}

interface RegisterData {
  email: string
  password: string
  username: string
  firstName?: string
  lastName?: string
}

// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Context
const AuthContext = createContext<AuthContextType | null>(null)

// Helper function to handle API requests with credentials
async function fetchWithCredentials(url: string, options: RequestInit = {}) {
  const response = await fetch(`${API_BASE_URL}${url}`, {
    ...options,
    credentials: 'include', // Always include cookies
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })
  
  return response
}

// CSRF token management
let csrfTokenPromise: Promise<string> | null = null

async function fetchCSRFToken(): Promise<string> {
  if (csrfTokenPromise) {
    return csrfTokenPromise
  }

  csrfTokenPromise = (async () => {
    try {
      const response = await fetchWithCredentials('/api/auth/csrf-token')
      if (!response.ok) {
        throw new Error('Failed to fetch CSRF token')
      }
      const data = await response.json()
      return data.csrfToken
    } catch (error) {
      csrfTokenPromise = null
      throw error
    }
  })()

  return csrfTokenPromise
}

export function SecureAuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [csrfToken, setCSRFToken] = useState<string | null>(null)
  const router = useRouter()
  const { identify, track } = useAnalytics()

  const isAuthenticated = !!user

  // Fetch CSRF token on mount
  useEffect(() => {
    fetchCSRFToken()
      .then(token => setCSRFToken(token))
      .catch(console.error)
  }, [])

  // Initialize auth state by checking session
  useEffect(() => {
    checkSession()
  }, [])

  // Set up session refresh interval
  useEffect(() => {
    if (!isAuthenticated) return

    // Refresh session every 14 minutes (assuming 15-minute token expiry)
    const refreshInterval = setInterval(() => {
      refreshSession()
    }, 14 * 60 * 1000)

    return () => clearInterval(refreshInterval)
  }, [isAuthenticated])

  const checkSession = async () => {
    try {
      const response = await fetchWithCredentials('/api/auth/session')
      
      if (response.ok) {
        const data = await response.json()
        if (data.user) {
          setUser(data.user)
          identify(data.user.id, {
            email: data.user.email,
            username: data.user.username,
            tier: data.user.tier,
            isVerified: data.user.isVerified,
          })
        }
      }
    } catch (error) {
      console.error('Session check failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const login = async (email: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      setIsLoading(true)
      
      // Get fresh CSRF token
      const token = await fetchCSRFToken()
      
      const response = await fetchWithCredentials('/api/auth/login', {
        method: 'POST',
        headers: {
          'X-CSRF-Token': token,
        },
        body: JSON.stringify({ email, password }),
      })

      const data = await response.json()

      if (!response.ok) {
        return { success: false, error: data.message || 'Login failed' }
      }

      const { user: userData } = data
      
      // Set user state
      setUser(userData)
      
      // Track login event
      identify(userData.id, {
        email: userData.email,
        username: userData.username,
        tier: userData.tier,
      })
      
      track('user_login', {
        method: 'email',
        userId: userData.id,
        tier: userData.tier,
      })

      return { success: true }
    } catch (error) {
      console.error('Login error:', error)
      return { success: false, error: 'Network error. Please try again.' }
    } finally {
      setIsLoading(false)
    }
  }

  const register = async (userData: RegisterData): Promise<{ success: boolean; error?: string }> => {
    try {
      setIsLoading(true)
      
      // Get fresh CSRF token
      const token = await fetchCSRFToken()
      
      const response = await fetchWithCredentials('/api/auth/register', {
        method: 'POST',
        headers: {
          'X-CSRF-Token': token,
        },
        body: JSON.stringify(userData),
      })

      const data = await response.json()

      if (!response.ok) {
        return { success: false, error: data.message || 'Registration failed' }
      }

      track('user_register', {
        method: 'email',
        username: userData.username,
      })

      return { success: true }
    } catch (error) {
      console.error('Registration error:', error)
      return { success: false, error: 'Network error. Please try again.' }
    } finally {
      setIsLoading(false)
    }
  }

  const logout = async () => {
    try {
      // Get fresh CSRF token
      const token = await fetchCSRFToken()
      
      await fetchWithCredentials('/api/auth/logout', {
        method: 'POST',
        headers: {
          'X-CSRF-Token': token,
        },
      })

      track('user_logout', {
        userId: user?.id,
      })
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      // Clear local state regardless of server response
      setUser(null)
      router.push('/')
    }
  }

  const refreshSession = useCallback(async () => {
    try {
      const response = await fetchWithCredentials('/api/auth/refresh', {
        method: 'POST',
      })

      if (!response.ok) {
        // Session expired, logout
        await logout()
      }
    } catch (error) {
      console.error('Session refresh failed:', error)
      await logout()
    }
  }, [])

  const updateUser = async (updates: Partial<User>) => {
    if (!user) return

    try {
      const token = await fetchCSRFToken()
      
      const response = await fetchWithCredentials('/api/user/profile', {
        method: 'PATCH',
        headers: {
          'X-CSRF-Token': token,
        },
        body: JSON.stringify(updates),
      })

      if (response.ok) {
        const updatedUser = await response.json()
        setUser(updatedUser)
        
        track('user_profile_update', {
          userId: user.id,
          fields: Object.keys(updates),
        })
      }
    } catch (error) {
      console.error('User update failed:', error)
      throw error
    }
  }

  const deleteAccount = async () => {
    if (!user) return

    try {
      const token = await fetchCSRFToken()
      
      const response = await fetchWithCredentials('/api/user/delete', {
        method: 'DELETE',
        headers: {
          'X-CSRF-Token': token,
        },
      })

      if (response.ok) {
        track('user_delete_account', {
          userId: user.id,
        })
        
        await logout()
      }
    } catch (error) {
      console.error('Account deletion failed:', error)
      throw error
    }
  }

  const resendVerification = async () => {
    if (!user) return

    try {
      const token = await fetchCSRFToken()
      
      await fetchWithCredentials('/api/auth/resend-verification', {
        method: 'POST',
        headers: {
          'X-CSRF-Token': token,
        },
        body: JSON.stringify({ email: user.email }),
      })

      track('verification_email_resent', {
        userId: user.id,
      })
    } catch (error) {
      console.error('Resend verification failed:', error)
      throw error
    }
  }

  const requestPasswordReset = async (email: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const token = await fetchCSRFToken()
      
      const response = await fetchWithCredentials('/api/auth/password-reset-request', {
        method: 'POST',
        headers: {
          'X-CSRF-Token': token,
        },
        body: JSON.stringify({ email }),
      })

      const data = await response.json()

      if (!response.ok) {
        return { success: false, error: data.message || 'Password reset request failed' }
      }

      track('password_reset_requested', { email })
      return { success: true }
    } catch (error) {
      console.error('Password reset request error:', error)
      return { success: false, error: 'Network error. Please try again.' }
    }
  }

  const resetPassword = async (resetToken: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const csrfToken = await fetchCSRFToken()
      
      const response = await fetchWithCredentials('/api/auth/password-reset', {
        method: 'POST',
        headers: {
          'X-CSRF-Token': csrfToken,
        },
        body: JSON.stringify({ token: resetToken, password }),
      })

      const data = await response.json()

      if (!response.ok) {
        return { success: false, error: data.message || 'Password reset failed' }
      }

      track('password_reset_completed')
      return { success: true }
    } catch (error) {
      console.error('Password reset error:', error)
      return { success: false, error: 'Network error. Please try again.' }
    }
  }

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated,
    csrfToken,
    login,
    register,
    logout,
    refreshSession,
    updateUser,
    deleteAccount,
    resendVerification,
    requestPasswordReset,
    resetPassword,
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within a SecureAuthProvider')
  }
  return context
}

// HOC for protected routes
export function withAuth<P extends object>(
  Component: React.ComponentType<P>,
  redirectTo: string = '/login'
) {
  return function AuthenticatedComponent(props: P) {
    const { isAuthenticated, isLoading } = useAuth()
    const router = useRouter()

    useEffect(() => {
      if (!isLoading && !isAuthenticated) {
        router.push(redirectTo)
      }
    }, [isAuthenticated, isLoading, router])

    if (isLoading) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-500"></div>
        </div>
      )
    }

    if (!isAuthenticated) {
      return null
    }

    return <Component {...props} />
  }
}