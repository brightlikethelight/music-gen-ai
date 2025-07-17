'use client'

import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useAnalytics } from '@/components/analytics/AnalyticsProvider'
import { apiClient, authAPI, userAPI } from '@/lib/api-client'
import { AuthMigrationService, AuthMigrationNotification } from '@/lib/auth-migration'

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

const AuthContext = createContext<AuthContextType | null>(null)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [csrfToken, setCSRFToken] = useState<string | null>(null)
  const [migrationComplete, setMigrationComplete] = useState(false)
  const router = useRouter()
  const { identify, track } = useAnalytics()

  const isAuthenticated = !!user

  // Check for migration on mount
  useEffect(() => {
    const checkMigration = async () => {
      if (AuthMigrationService.needsMigration()) {
        // Migration will be handled by AuthMigrationNotification
        setMigrationComplete(false)
      } else {
        setMigrationComplete(true)
      }
    }
    checkMigration()
  }, [])

  // Initialize auth state after migration
  useEffect(() => {
    if (migrationComplete) {
      checkSession()
    }
  }, [migrationComplete])

  // Set up session refresh interval
  useEffect(() => {
    if (!isAuthenticated) return

    const refreshInterval = setInterval(() => {
      refreshSession()
    }, 14 * 60 * 1000) // Refresh every 14 minutes (tokens expire in 15)

    return () => clearInterval(refreshInterval)
  }, [isAuthenticated])

  const checkSession = async () => {
    try {
      const session = await authAPI.getSession()
      
      if (session.user) {
        setUser(session.user)
        setCSRFToken(session.csrfToken)
        
        identify(session.user.id, {
          email: session.user.email,
          username: session.user.username,
          tier: session.user.tier,
          isVerified: session.user.isVerified,
        })
        
        // Update last login
        await userAPI.updateLastLogin()
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
      
      const data = await authAPI.login(email, password)

      if (data.user) {
        setUser(data.user)
        setCSRFToken(data.csrfToken)
        
        // Track login event
        identify(data.user.id, {
          email: data.user.email,
          username: data.user.username,
          tier: data.user.tier,
        })
        
        track('user_login', {
          method: 'email',
          userId: data.user.id,
          tier: data.user.tier,
        })

        return { success: true }
      }

      return { success: false, error: 'Login failed' }
    } catch (error: any) {
      console.error('Login error:', error)
      return { success: false, error: error.message || 'Network error. Please try again.' }
    } finally {
      setIsLoading(false)
    }
  }

  const register = async (userData: RegisterData): Promise<{ success: boolean; error?: string }> => {
    try {
      setIsLoading(true)
      
      const data = await authAPI.register(userData)

      track('user_register', {
        method: 'email',
        username: userData.username,
      })

      return { success: true }
    } catch (error: any) {
      console.error('Registration error:', error)
      return { success: false, error: error.message || 'Network error. Please try again.' }
    } finally {
      setIsLoading(false)
    }
  }

  const logout = async () => {
    try {
      await authAPI.logout()

      track('user_logout', {
        userId: user?.id,
      })
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      // Clear local state regardless of server response
      setUser(null)
      setCSRFToken(null)
      apiClient.clearCSRFToken()
      router.push('/')
    }
  }

  const refreshSession = useCallback(async () => {
    try {
      await authAPI.refreshSession()
      // Session cookies will be automatically updated
    } catch (error) {
      console.error('Session refresh failed:', error)
      await logout()
    }
  }, [])

  const updateUser = async (updates: Partial<User>) => {
    if (!user) return

    try {
      const updatedUser = await userAPI.updateProfile(updates)
      setUser(updatedUser)
      
      track('user_profile_update', {
        userId: user.id,
        fields: Object.keys(updates),
      })
    } catch (error) {
      console.error('User update failed:', error)
      throw error
    }
  }

  const deleteAccount = async () => {
    if (!user) return

    try {
      await userAPI.deleteAccount()
      
      track('user_delete_account', {
        userId: user.id,
      })
      
      await logout()
    } catch (error) {
      console.error('Account deletion failed:', error)
      throw error
    }
  }

  const resendVerification = async () => {
    if (!user) return

    try {
      await authAPI.resendVerification(user.email)

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
      await authAPI.requestPasswordReset(email)

      track('password_reset_requested', { email })
      return { success: true }
    } catch (error: any) {
      console.error('Password reset request error:', error)
      return { success: false, error: error.message || 'Network error. Please try again.' }
    }
  }

  const resetPassword = async (token: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      await authAPI.resetPassword(token, password)

      track('password_reset_completed')
      return { success: true }
    } catch (error: any) {
      console.error('Password reset error:', error)
      return { success: false, error: error.message || 'Network error. Please try again.' }
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
      <AuthMigrationNotification />
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
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