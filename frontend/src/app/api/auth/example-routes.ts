/**
 * Example Next.js API routes for cookie-based authentication
 * These routes act as a proxy to the backend API and handle cookie management
 */

import { NextRequest, NextResponse } from 'next/server'

const API_BASE_URL = process.env.BACKEND_API_URL || 'http://localhost:8000'

// Helper to forward cookies
function forwardCookies(backendResponse: Response, nextResponse: NextResponse) {
  // Forward Set-Cookie headers from backend to client
  const setCookieHeaders = backendResponse.headers.get('set-cookie')
  if (setCookieHeaders) {
    // Parse and set cookies properly
    const cookies = setCookieHeaders.split(',').map(c => c.trim())
    cookies.forEach(cookie => {
      const [nameValue, ...attributes] = cookie.split(';')
      const [name, value] = nameValue.split('=')
      
      const options: any = {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax' as const,
        path: '/',
      }
      
      // Parse cookie attributes
      attributes.forEach(attr => {
        const [key, val] = attr.trim().split('=')
        if (key.toLowerCase() === 'max-age') {
          options.maxAge = parseInt(val)
        } else if (key.toLowerCase() === 'expires') {
          options.expires = new Date(val)
        }
      })
      
      nextResponse.cookies.set(name.trim(), value.trim(), options)
    })
  }
}

// Login endpoint
export async function POST(request: NextRequest) {
  if (request.nextUrl.pathname === '/api/auth/login') {
    try {
      const body = await request.json()
      
      // Forward login request to backend
      const backendResponse = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        credentials: 'include',
      })
      
      const data = await backendResponse.json()
      
      if (!backendResponse.ok) {
        return NextResponse.json(data, { status: backendResponse.status })
      }
      
      // Create response and forward cookies
      const response = NextResponse.json(data)
      forwardCookies(backendResponse, response)
      
      return response
    } catch (error) {
      return NextResponse.json(
        { error: 'Internal server error' },
        { status: 500 }
      )
    }
  }
  
  // Logout endpoint
  if (request.nextUrl.pathname === '/api/auth/logout') {
    try {
      // Get cookies from request
      const cookieHeader = request.headers.get('cookie') || ''
      
      // Forward logout request to backend
      const backendResponse = await fetch(`${API_BASE_URL}/api/auth/logout`, {
        method: 'POST',
        headers: {
          'Cookie': cookieHeader,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      })
      
      // Clear cookies regardless of backend response
      const response = NextResponse.json({ success: true })
      
      // Clear auth cookies
      response.cookies.set('auth_token', '', {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        maxAge: 0,
        path: '/',
      })
      
      response.cookies.set('refresh_token', '', {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        maxAge: 0,
        path: '/',
      })
      
      response.cookies.set('csrf_token', '', {
        httpOnly: false,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        maxAge: 0,
        path: '/',
      })
      
      return response
    } catch (error) {
      return NextResponse.json(
        { error: 'Internal server error' },
        { status: 500 }
      )
    }
  }
}

// Session check endpoint
export async function GET(request: NextRequest) {
  if (request.nextUrl.pathname === '/api/auth/session') {
    try {
      // Get cookies from request
      const cookieHeader = request.headers.get('cookie') || ''
      
      // Forward session check to backend
      const backendResponse = await fetch(`${API_BASE_URL}/api/auth/session`, {
        method: 'GET',
        headers: {
          'Cookie': cookieHeader,
        },
        credentials: 'include',
      })
      
      const data = await backendResponse.json()
      
      return NextResponse.json(data, { status: backendResponse.status })
    } catch (error) {
      return NextResponse.json(
        { error: 'Internal server error' },
        { status: 500 }
      )
    }
  }
  
  // CSRF token endpoint
  if (request.nextUrl.pathname === '/api/auth/csrf-token') {
    try {
      // Get cookies from request
      const cookieHeader = request.headers.get('cookie') || ''
      
      // Forward CSRF token request to backend
      const backendResponse = await fetch(`${API_BASE_URL}/api/auth/csrf-token`, {
        method: 'GET',
        headers: {
          'Cookie': cookieHeader,
        },
        credentials: 'include',
      })
      
      const data = await backendResponse.json()
      
      // Optionally set CSRF token as a non-httpOnly cookie for easy access
      const response = NextResponse.json(data)
      
      if (data.csrfToken) {
        response.cookies.set('csrf_token', data.csrfToken, {
          httpOnly: false, // Accessible to JavaScript
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'lax',
          path: '/',
        })
      }
      
      return response
    } catch (error) {
      return NextResponse.json(
        { error: 'Internal server error' },
        { status: 500 }
      )
    }
  }
}

// Middleware to check authentication for protected API routes
export async function authMiddleware(request: NextRequest) {
  const authToken = request.cookies.get('auth_token')
  
  if (!authToken) {
    return NextResponse.json(
      { error: 'Authentication required' },
      { status: 401 }
    )
  }
  
  // Optionally verify token with backend
  // For better performance, you might want to verify JWT locally
  
  return null // Continue to route handler
}

// Example protected API route
export async function protectedRoute(request: NextRequest) {
  // Check authentication
  const authError = await authMiddleware(request)
  if (authError) return authError
  
  // Route logic here
  return NextResponse.json({ message: 'Protected data' })
}