'use client'

import { SessionProvider } from 'next-auth/react'
import { SWRConfig } from 'swr'
import { ThemeProvider } from 'next-themes'
import { AudioProvider } from '@/contexts/AudioContext'
import { GenerationProvider } from '@/contexts/GenerationContext'
import { WebSocketProvider } from '@/contexts/WebSocketContext'
import { UserProvider } from '@/contexts/UserContext'
import { AnalyticsProvider } from '@/components/analytics/AnalyticsProvider'
import { FeedbackWidget } from '@/components/feedback/FeedbackWidget'
import { fetcher } from '@/lib/api'

interface ProvidersProps {
  children: React.ReactNode
}

export function Providers({ children }: ProvidersProps) {
  return (
    <SessionProvider>
      <ThemeProvider
        attribute="class"
        defaultTheme="system"
        enableSystem
        disableTransitionOnChange
      >
        <SWRConfig
          value={{
            fetcher,
            revalidateOnFocus: false,
            revalidateOnReconnect: true,
            shouldRetryOnError: true,
            errorRetryCount: 3,
            errorRetryInterval: 1000,
            dedupingInterval: 2000,
            focusThrottleInterval: 5000,
            loadingTimeout: 3000,
            onError: (error, key) => {
              if (error.status !== 403 && error.status !== 404) {
                console.error('SWR Error:', error, 'Key:', key)
              }
            },
            onErrorRetry: (error, key, config, revalidate, { retryCount }) => {
              // Don't retry on 404
              if (error.status === 404) return
              
              // Don't retry on 401/403 (auth errors)
              if (error.status === 401 || error.status === 403) return
              
              // Only retry up to 3 times
              if (retryCount >= 3) return
              
              // Retry after exponential backoff
              setTimeout(() => revalidate({ retryCount }), Math.pow(2, retryCount) * 1000)
            },
          }}
        >
          <AnalyticsProvider config={{ enableUserTesting: true, debug: process.env.NODE_ENV === 'development' }}>
            <UserProvider>
              <WebSocketProvider>
                <AudioProvider>
                  <GenerationProvider>
                    {children}
                    <FeedbackWidget />
                  </GenerationProvider>
                </AudioProvider>
              </WebSocketProvider>
            </UserProvider>
          </AnalyticsProvider>
        </SWRConfig>
      </ThemeProvider>
    </SessionProvider>
  )
}