/**
 * Mock Service Worker (MSW) server configuration for testing
 * 
 * Provides mock API responses for all backend endpoints used in tests.
 */

import { setupServer } from 'msw/node';
import { rest } from 'msw';
import { mockApiResponses, mockUser } from './index';

// Mock handlers for all API endpoints
export const handlers = [
  // Authentication endpoints
  rest.post('/api/auth/login', (req, res, ctx) => {
    const { email, password } = req.body as any;
    
    if (email === 'test@example.com' && password === 'password123') {
      return res(
        ctx.status(200),
        ctx.json(mockApiResponses.loginSuccess),
        ctx.set('Set-Cookie', 'auth_token=mock-token; HttpOnly; Secure; SameSite=Strict')
      );
    }
    
    return res(
      ctx.status(401),
      ctx.json(mockApiResponses.loginError)
    );
  }),

  rest.post('/api/auth/register', (req, res, ctx) => {
    const { email, username, password } = req.body as any;
    
    // Simulate validation errors
    if (!email || !username || !password) {
      return res(
        ctx.status(422),
        ctx.json({ detail: 'All fields are required' })
      );
    }
    
    if (email === 'existing@example.com') {
      return res(
        ctx.status(400),
        ctx.json({ detail: 'Email already exists' })
      );
    }
    
    return res(
      ctx.status(201),
      ctx.json(mockApiResponses.registerSuccess),
      ctx.set('Set-Cookie', 'auth_token=mock-token; HttpOnly; Secure; SameSite=Strict')
    );
  }),

  rest.post('/api/auth/logout', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({ message: 'Logged out successfully' }),
      ctx.set('Set-Cookie', 'auth_token=; HttpOnly; Secure; SameSite=Strict; Max-Age=0')
    );
  }),

  rest.get('/api/auth/session', (req, res, ctx) => {
    const authHeader = req.headers.get('Authorization');
    const hasCookie = req.headers.get('Cookie')?.includes('auth_token=mock-token');
    
    if (authHeader?.includes('Bearer mock-token') || hasCookie) {
      return res(
        ctx.status(200),
        ctx.json({
          authenticated: true,
          user: mockUser,
          csrfToken: 'mock-csrf-token',
        })
      );
    }
    
    return res(
      ctx.status(200),
      ctx.json({ authenticated: false })
    );
  }),

  rest.get('/api/auth/csrf-token', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({ csrfToken: 'mock-csrf-token' }),
      ctx.set('Set-Cookie', 'csrf_token=mock-csrf-token; HttpOnly; Secure; SameSite=Strict')
    );
  }),

  rest.post('/api/auth/refresh', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        message: 'Tokens refreshed successfully',
      })
    );
  }),

  // Health check endpoints
  rest.get('/health', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        status: 'healthy',
        uptime: 12345,
        timestamp: new Date().toISOString(),
      })
    );
  }),

  rest.get('/health/detailed', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        status: 'healthy',
        model_manager: { status: 'loaded' },
        memory_usage: 45.2,
        disk_usage: 23.1,
      })
    );
  }),

  // Music generation endpoints
  rest.post('/api/v1/generate/', (req, res, ctx) => {
    const { prompt, duration } = req.body as any;
    
    // Simulate validation errors
    if (!prompt || prompt.trim().length === 0) {
      return res(
        ctx.status(422),
        ctx.json({ detail: 'Prompt is required' })
      );
    }
    
    if (duration && (duration < 1 || duration > 60)) {
      return res(
        ctx.status(422),
        ctx.json({ detail: 'Duration must be between 1 and 60 seconds' })
      );
    }
    
    // Simulate rate limiting
    if (prompt.includes('rate-limit-test')) {
      return res(
        ctx.status(429),
        ctx.json({ detail: 'Rate limit exceeded' }),
        ctx.set('Retry-After', '60')
      );
    }
    
    return res(
      ctx.status(200),
      ctx.json(mockApiResponses.generationRequest)
    );
  }),

  rest.post('/api/v1/generate/batch', (req, res, ctx) => {
    const { requests } = req.body as any;
    
    if (!requests || !Array.isArray(requests) || requests.length === 0) {
      return res(
        ctx.status(422),
        ctx.json({ detail: 'Requests array is required' })
      );
    }
    
    if (requests.length > 5) {
      return res(
        ctx.status(422),
        ctx.json({ detail: 'Maximum 5 requests per batch' })
      );
    }
    
    return res(
      ctx.status(200),
      ctx.json({
        batch_id: 'batch-123',
        task_ids: requests.map((_, i) => `task-batch-${i + 1}`),
        total_requests: requests.length,
        status: 'pending',
      })
    );
  }),

  rest.get('/api/v1/generate/:taskId', (req, res, ctx) => {
    const { taskId } = req.params;
    
    if (taskId === 'task-not-found') {
      return res(
        ctx.status(404),
        ctx.json({ detail: 'Task not found' })
      );
    }
    
    if (taskId === 'task-failed') {
      return res(
        ctx.status(200),
        ctx.json(mockApiResponses.generationError)
      );
    }
    
    return res(
      ctx.status(200),
      ctx.json(mockApiResponses.generationStatus)
    );
  }),

  rest.get('/api/v1/generate/batch/:batchId', (req, res, ctx) => {
    const { batchId } = req.params;
    
    return res(
      ctx.status(200),
      ctx.json({
        batch_id: batchId,
        total: 3,
        completed: 2,
        failed: 1,
        pending: 0,
        tasks: [
          { task_id: 'task-1', status: 'completed' },
          { task_id: 'task-2', status: 'completed' },
          { task_id: 'task-3', status: 'failed' },
        ],
      })
    );
  }),

  // File download endpoints
  rest.get('/download/:taskId', (req, res, ctx) => {
    const { taskId } = req.params;
    
    if (taskId === 'task-not-found') {
      return res(
        ctx.status(404),
        ctx.json({ detail: 'Task not found' })
      );
    }
    
    if (taskId === 'task-not-completed') {
      return res(
        ctx.status(400),
        ctx.json({ detail: 'Generation not completed' })
      );
    }
    
    // Return mock audio data
    const mockAudioData = new ArrayBuffer(1024); // 1KB of mock audio
    return res(
      ctx.status(200),
      ctx.set('Content-Type', 'audio/wav'),
      ctx.set('Content-Length', '1024'),
      ctx.set('Content-Disposition', `attachment; filename="music-${taskId}.wav"`),
      ctx.body(mockAudioData)
    );
  }),

  // Streaming endpoints
  rest.post('/api/v1/stream/session', (req, res, ctx) => {
    const { prompt, duration } = req.body as any;
    
    if (!prompt) {
      return res(
        ctx.status(422),
        ctx.json({ detail: 'Prompt is required' })
      );
    }
    
    return res(
      ctx.status(200),
      ctx.json({
        session_id: 'stream-session-123',
        websocket_url: '/api/v1/stream/ws/stream-session-123',
      })
    );
  }),

  rest.get('/api/v1/stream/sessions', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        sessions: [
          {
            session_id: 'session1',
            prompt: 'rock music',
            duration: 10.0,
            progress: 0.5,
            status: 'streaming',
            created_at: '2024-01-01T12:00:00Z',
          },
        ],
        count: 1,
        max_sessions: 10,
      })
    );
  }),

  rest.delete('/api/v1/stream/session/:sessionId', (req, res, ctx) => {
    const { sessionId } = req.params;
    
    if (sessionId === 'nonexistent-session') {
      return res(
        ctx.status(404),
        ctx.json({ detail: 'Session not found' })
      );
    }
    
    return res(
      ctx.status(200),
      ctx.json({
        message: 'Session stopped',
        session_id: sessionId,
      })
    );
  }),

  // Project management endpoints
  rest.get('/api/v1/projects/', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        projects: mockApiResponses.projects,
        total: mockApiResponses.projects.length,
      })
    );
  }),

  rest.post('/api/v1/projects/', (req, res, ctx) => {
    const { name, description } = req.body as any;
    
    if (!name || name.trim().length === 0) {
      return res(
        ctx.status(422),
        ctx.json({ detail: 'Project name is required' })
      );
    }
    
    return res(
      ctx.status(201),
      ctx.json(mockApiResponses.createProject)
    );
  }),

  rest.get('/api/v1/projects/:projectId', (req, res, ctx) => {
    const { projectId } = req.params;
    
    const project = mockApiResponses.projects.find(p => p.id === projectId);
    if (!project) {
      return res(
        ctx.status(404),
        ctx.json({ detail: 'Project not found' })
      );
    }
    
    return res(
      ctx.status(200),
      ctx.json(project)
    );
  }),

  rest.put('/api/v1/projects/:projectId', (req, res, ctx) => {
    const { projectId } = req.params;
    const { name, description } = req.body as any;
    
    const project = mockApiResponses.projects.find(p => p.id === projectId);
    if (!project) {
      return res(
        ctx.status(404),
        ctx.json({ detail: 'Project not found' })
      );
    }
    
    return res(
      ctx.status(200),
      ctx.json({
        ...project,
        name: name || project.name,
        description: description || project.description,
        updated_at: new Date().toISOString(),
      })
    );
  }),

  rest.delete('/api/v1/projects/:projectId', (req, res, ctx) => {
    const { projectId } = req.params;
    
    const project = mockApiResponses.projects.find(p => p.id === projectId);
    if (!project) {
      return res(
        ctx.status(404),
        ctx.json({ detail: 'Project not found' })
      );
    }
    
    return res(
      ctx.status(204)
    );
  }),

  // User profile endpoints
  rest.get('/api/v1/user/profile', (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        user: mockUser,
        settings: {
          theme: 'light',
          notifications: true,
          autoSave: true,
        },
      })
    );
  }),

  rest.put('/api/v1/user/profile', (req, res, ctx) => {
    const { username, email } = req.body as any;
    
    return res(
      ctx.status(200),
      ctx.json({
        ...mockUser,
        username: username || mockUser.username,
        email: email || mockUser.email,
      })
    );
  }),

  // Error simulation endpoints for testing
  rest.get('/api/test/error/:errorCode', (req, res, ctx) => {
    const { errorCode } = req.params;
    const code = parseInt(errorCode as string);
    
    return res(
      ctx.status(code),
      ctx.json({ detail: `Simulated ${code} error` })
    );
  }),

  rest.get('/api/test/timeout', (req, res, ctx) => {
    // Simulate timeout by delaying response
    return res(
      ctx.delay(30000), // 30 second delay
      ctx.status(200),
      ctx.json({ message: 'This should timeout' })
    );
  }),

  // Catch-all handler for unhandled requests
  rest.all('*', (req, res, ctx) => {
    console.warn(`Unhandled request: ${req.method} ${req.url.href}`);
    return res(
      ctx.status(404),
      ctx.json({ detail: 'Endpoint not found' })
    );
  }),
];

// Setup MSW server
export const server = setupServer(...handlers);

// Helper function to override handlers for specific tests
export const overrideHandler = (handler: any) => {
  server.use(handler);
};

// Helper function to simulate network errors
export const simulateNetworkError = (url: string) => {
  server.use(
    rest.all(url, (req, res, ctx) => {
      return res.networkError('Network error');
    })
  );
};

// Helper function to simulate server errors
export const simulateServerError = (url: string, status = 500) => {
  server.use(
    rest.all(url, (req, res, ctx) => {
      return res(
        ctx.status(status),
        ctx.json({ detail: 'Internal server error' })
      );
    })
  );
};

// Helper function to simulate rate limiting
export const simulateRateLimit = (url: string) => {
  server.use(
    rest.all(url, (req, res, ctx) => {
      return res(
        ctx.status(429),
        ctx.json({ detail: 'Rate limit exceeded' }),
        ctx.set('Retry-After', '60')
      );
    })
  );
};