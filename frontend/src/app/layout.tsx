import { Inter, Poppins, JetBrains_Mono } from 'next/font/google'
import { Metadata } from 'next'
import { Providers } from './providers'
import { ResponsiveLayout } from '@/components/layout/ResponsiveLayout'
import { Footer } from '@/components/layout/Footer'
import { Toaster } from 'react-hot-toast'
import { ProgressBar } from '@/components/ui/ProgressBar'
import '@/styles/globals.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

const poppins = Poppins({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  variable: '--font-poppins',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: {
    default: 'MusicGen AI - Professional Music Generation Platform',
    template: '%s | MusicGen AI',
  },
  description: 'Create professional-quality music with AI. Generate, edit, and share your compositions with our advanced music generation platform.',
  keywords: [
    'AI music generation',
    'music composition',
    'audio editing',
    'music production',
    'artificial intelligence',
    'sound design',
    'music technology',
  ],
  authors: [{ name: 'MusicGen AI Team' }],
  creator: 'MusicGen AI',
  publisher: 'MusicGen AI',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
    siteName: 'MusicGen AI',
    title: 'MusicGen AI - Professional Music Generation Platform',
    description: 'Create professional-quality music with AI. Generate, edit, and share your compositions.',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'MusicGen AI Platform',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'MusicGen AI - Professional Music Generation Platform',
    description: 'Create professional-quality music with AI. Generate, edit, and share your compositions.',
    images: ['/og-image.jpg'],
    creator: '@musicgenai',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
  manifest: '/site.webmanifest',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${poppins.variable} ${jetbrainsMono.variable}`}>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#0ea5e9" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
      </head>
      <body className="min-h-screen bg-gray-900 font-sans antialiased">
        <Providers>
          <ProgressBar />
          <ResponsiveLayout>
            {children}
          </ResponsiveLayout>
          
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#ffffff',
                color: '#374151',
                border: '1px solid #e5e7eb',
                borderRadius: '0.5rem',
                fontSize: '0.875rem',
                maxWidth: '500px',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#ffffff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#ffffff',
                },
              },
            }}
          />
        </Providers>
      </body>
    </html>
  )
}