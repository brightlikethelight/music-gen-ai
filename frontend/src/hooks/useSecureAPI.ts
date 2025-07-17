/**
 * Custom hook for making secure API calls with httpOnly cookies and CSRF protection
 */

import { useState, useCallback } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { apiClient } from '@/lib/api-client'

interface UseSecureAPIOptions {
  onSuccess?: (data: any) => void
  onError?: (error: Error) => void
}

export function useSecureAPI<T = any>(
  endpoint: string,
  options?: UseSecureAPIOptions
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const { csrfToken } = useAuth()

  const execute = useCallback(
    async (method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' = 'GET', body?: any) => {
      setLoading(true)
      setError(null)

      try {
        let result: T

        switch (method) {
          case 'GET':
            result = await apiClient.get<T>(endpoint)
            break
          case 'POST':
            result = await apiClient.post<T>(endpoint, body)
            break
          case 'PUT':
            result = await apiClient.put<T>(endpoint, body)
            break
          case 'PATCH':
            result = await apiClient.patch<T>(endpoint, body)
            break
          case 'DELETE':
            result = await apiClient.delete<T>(endpoint)
            break
          default:
            throw new Error(`Unsupported method: ${method}`)
        }

        setData(result)
        options?.onSuccess?.(result)
        return result
      } catch (err) {
        const error = err as Error
        setError(error)
        options?.onError?.(error)
        throw error
      } finally {
        setLoading(false)
      }
    },
    [endpoint, options]
  )

  return {
    data,
    loading,
    error,
    execute,
    get: () => execute('GET'),
    post: (body?: any) => execute('POST', body),
    put: (body?: any) => execute('PUT', body),
    patch: (body?: any) => execute('PATCH', body),
    delete: () => execute('DELETE'),
  }
}

// Example usage in components
export function ExampleComponent() {
  const { data, loading, error, post } = useSecureAPI('/api/v1/generate', {
    onSuccess: (result) => {
      console.log('Generation started:', result)
    },
    onError: (error) => {
      console.error('Generation failed:', error)
    },
  })

  const handleGenerate = async () => {
    try {
      await post({
        prompt: 'Create upbeat jazz music',
        duration: 30,
      })
    } catch (error) {
      // Error already handled by onError callback
    }
  }

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>

  return (
    <div>
      <button onClick={handleGenerate}>Generate Music</button>
      {data && <div>Task ID: {data.taskId}</div>}
    </div>
  )
}

// Hook for file uploads with CSRF protection
export function useSecureFileUpload(endpoint: string) {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<Error | null>(null)
  const { csrfToken } = useAuth()

  const upload = useCallback(
    async (file: File, additionalData?: Record<string, any>) => {
      setUploading(true)
      setError(null)
      setProgress(0)

      try {
        const formData = new FormData()
        formData.append('file', file)
        
        // Add additional data if provided
        if (additionalData) {
          Object.entries(additionalData).forEach(([key, value]) => {
            formData.append(key, value)
          })
        }

        // Get CSRF token
        const token = await apiClient['getCSRFToken']()

        const xhr = new XMLHttpRequest()

        // Track upload progress
        xhr.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            const percentComplete = Math.round((event.loaded / event.total) * 100)
            setProgress(percentComplete)
          }
        })

        // Handle completion
        return new Promise((resolve, reject) => {
          xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              try {
                const response = JSON.parse(xhr.responseText)
                resolve(response)
              } catch (e) {
                resolve(xhr.responseText)
              }
            } else {
              reject(new Error(`Upload failed: ${xhr.statusText}`))
            }
          })

          xhr.addEventListener('error', () => {
            reject(new Error('Upload failed'))
          })

          // Configure request
          xhr.open('POST', `${process.env.NEXT_PUBLIC_API_URL || ''}${endpoint}`)
          xhr.setRequestHeader('X-CSRF-Token', token)
          xhr.withCredentials = true // Include cookies

          // Send request
          xhr.send(formData)
        })
      } catch (err) {
        const error = err as Error
        setError(error)
        throw error
      } finally {
        setUploading(false)
      }
    },
    [endpoint]
  )

  return {
    upload,
    uploading,
    progress,
    error,
  }
}

// Hook for real-time updates with authentication
export function useSecureWebSocket(url: string) {
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<any[]>([])
  const { user } = useAuth()

  useEffect(() => {
    if (!user) return

    // Include auth token in WebSocket connection
    const wsUrl = new URL(url)
    wsUrl.searchParams.append('token', 'from-cookie') // Backend should read from cookie

    const ws = new WebSocket(wsUrl.toString())

    ws.onopen = () => {
      setConnected(true)
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setMessages((prev) => [...prev, data])
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setConnected(false)
    }

    ws.onclose = () => {
      setConnected(false)
      console.log('WebSocket disconnected')
    }

    return () => {
      ws.close()
    }
  }, [url, user])

  return {
    connected,
    messages,
  }
}

// Migration helper for updating old code
export function migrateAPICall(oldCode: string): string {
  // This is a helper function to show how to migrate old API calls
  
  // Example transformations:
  const migrations = {
    // Old: fetch with Authorization header
    [`headers: {
      'Authorization': \`Bearer \${token}\`
    }`]: `credentials: 'include',
    headers: {
      'X-CSRF-Token': csrfToken
    }`,
    
    // Old: axios with token
    [`axios.defaults.headers.common['Authorization'] = \`Bearer \${token}\``]: 
    `// Token now handled via httpOnly cookies
    axios.defaults.withCredentials = true`,
    
    // Old: localStorage token check
    [`const token = localStorage.getItem('auth_token')`]: 
    `// Token now in httpOnly cookie, check session instead
    const { user } = useAuth()`,
  }

  let migratedCode = oldCode
  
  Object.entries(migrations).forEach(([old, replacement]) => {
    migratedCode = migratedCode.replace(old, replacement)
  })
  
  return migratedCode
}

import { useEffect } from 'react'