/**
 * Application Integration Tests
 * 
 * End-to-end integration tests for the complete Music Generation AI application.
 * These tests verify that all components work together correctly in real-world scenarios.
 */

import React from 'react';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../test-utils';
import { server } from '../test-utils/msw-server';
import { rest } from 'msw';
import App from '../app/page';

// Mock environment variables
process.env.NEXT_PUBLIC_API_URL = 'http://localhost:3000';
process.env.NEXT_PUBLIC_WS_URL = 'ws://localhost:3000';

// Enhanced mock handlers for complete application flow
const handlers = [
  // Authentication
  rest.post('/api/auth/login', async (req, res, ctx) => {
    const { email, password } = await req.json();
    if (email === 'test@example.com' && password === 'password123') {
      return res(
        ctx.json({
          user: { id: '1', email, name: 'Test User' },
          token: 'mock-jwt-token',
        })
      );
    }
    return res(ctx.status(401), ctx.json({ error: 'Invalid credentials' }));
  }),

  rest.post('/api/auth/register', async (req, res, ctx) => {
    const { email } = await req.json();
    return res(
      ctx.json({
        user: { id: '2', email, name: 'New User' },
        token: 'mock-jwt-token',
      })
    );
  }),

  rest.post('/api/auth/logout', (req, res, ctx) => {
    return res(ctx.json({ success: true }));
  }),

  // User profile
  rest.get('/api/user/profile', (req, res, ctx) => {
    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      return res(ctx.status(401));
    }
    return res(
      ctx.json({
        id: '1',
        email: 'test@example.com',
        name: 'Test User',
        subscription: 'pro',
        credits: 100,
      })
    );
  }),

  // Projects
  rest.get('/api/projects', (req, res, ctx) => {
    return res(
      ctx.json({
        projects: [
          { id: '1', name: 'Jazz Project', createdAt: new Date().toISOString() },
          { id: '2', name: 'Rock Project', createdAt: new Date().toISOString() },
        ],
      })
    );
  }),

  rest.post('/api/projects', async (req, res, ctx) => {
    const { name } = await req.json();
    return res(
      ctx.json({
        id: '3',
        name,
        createdAt: new Date().toISOString(),
      })
    );
  }),

  // Music generation
  rest.post('/api/generate', async (req, res, ctx) => {
    const { prompt } = await req.json();
    return res(
      ctx.delay(1000),
      ctx.json({
        id: 'gen-123',
        audioUrl: '/audio/generated.wav',
        duration: 30,
        prompt,
      })
    );
  }),

  rest.get('/api/generation/:id', (req, res, ctx) => {
    return res(
      ctx.json({
        id: req.params.id,
        status: 'completed',
        audioUrl: '/audio/generated.wav',
        waveformData: Array(1000).fill(0).map(() => Math.random()),
      })
    );
  }),

  // Community features
  rest.get('/api/community/trending', (req, res, ctx) => {
    return res(
      ctx.json({
        tracks: [
          {
            id: '1',
            title: 'Summer Vibes',
            artist: 'User123',
            likes: 245,
            plays: 1200,
            audioUrl: '/audio/summer-vibes.wav',
          },
          {
            id: '2',
            title: 'Night Drive',
            artist: 'User456',
            likes: 189,
            plays: 890,
            audioUrl: '/audio/night-drive.wav',
          },
        ],
      })
    );
  }),
];

describe('Complete Application Flow', () => {
  const user = userEvent.setup();

  beforeAll(() => {
    server.listen();
  });

  afterEach(() => {
    server.resetHandlers();
    localStorage.clear();
    sessionStorage.clear();
  });

  afterAll(() => {
    server.close();
  });

  describe('New User Journey', () => {
    it('should complete full onboarding and first generation flow', async () => {
      renderWithProviders(<App />);

      // 1. Landing page
      expect(screen.getByText(/create amazing music/i)).toBeInTheDocument();

      // 2. Click Get Started
      await user.click(screen.getByRole('button', { name: /get started/i }));

      // 3. Register form
      await waitFor(() => {
        expect(screen.getByPlaceholderText(/email/i)).toBeInTheDocument();
      });

      await user.type(screen.getByPlaceholderText(/email/i), 'newuser@example.com');
      await user.type(screen.getByPlaceholderText(/^password/i), 'SecurePass123!');
      await user.type(screen.getByPlaceholderText(/confirm password/i), 'SecurePass123!');
      await user.type(screen.getByPlaceholderText(/display name/i), 'New User');

      await user.click(screen.getByRole('button', { name: /register/i }));

      // 4. Dashboard after registration
      await waitFor(() => {
        expect(screen.getByText(/welcome, new user/i)).toBeInTheDocument();
      });

      // 5. Create first project
      await user.click(screen.getByRole('button', { name: /new project/i }));

      const modal = screen.getByRole('dialog');
      await user.type(within(modal).getByPlaceholderText(/project name/i), 'My First Song');
      await user.click(within(modal).getByRole('button', { name: /create/i }));

      // 6. Generation studio
      await waitFor(() => {
        expect(screen.getByPlaceholderText(/describe your music/i)).toBeInTheDocument();
      });

      // 7. Generate music
      await user.type(
        screen.getByPlaceholderText(/describe your music/i),
        'Upbeat electronic dance music with catchy melody'
      );

      await user.click(screen.getByRole('button', { name: /generate/i }));

      // 8. Wait for generation
      await waitFor(() => {
        expect(screen.getByText(/generating/i)).toBeInTheDocument();
      });

      // 9. View result
      await waitFor(() => {
        expect(screen.getByTestId('audio-player')).toBeInTheDocument();
        expect(screen.getByTestId('waveform')).toBeInTheDocument();
      }, { timeout: 5000 });

      // 10. Save to project
      await user.click(screen.getByRole('button', { name: /save to project/i }));

      await waitFor(() => {
        expect(screen.getByText(/saved successfully/i)).toBeInTheDocument();
      });
    });
  });

  describe('Returning User Workflow', () => {
    it('should handle login and resume work on existing project', async () => {
      // Pre-populate localStorage with auth token
      localStorage.setItem('authToken', 'existing-token');

      renderWithProviders(<App />);

      // Should load dashboard directly
      await waitFor(() => {
        expect(screen.getByText(/your projects/i)).toBeInTheDocument();
      });

      // View existing projects
      expect(screen.getByText('Jazz Project')).toBeInTheDocument();
      expect(screen.getByText('Rock Project')).toBeInTheDocument();

      // Open existing project
      await user.click(screen.getByText('Jazz Project'));

      // Should show project workspace
      await waitFor(() => {
        expect(screen.getByText(/jazz project/i)).toBeInTheDocument();
        expect(screen.getByTestId('project-workspace')).toBeInTheDocument();
      });

      // Continue generation
      await user.click(screen.getByRole('button', { name: /new generation/i }));

      await user.type(
        screen.getByPlaceholderText(/describe your music/i),
        'Smooth jazz with saxophone'
      );

      await user.click(screen.getByRole('button', { name: /generate/i }));

      await waitFor(() => {
        expect(screen.getByTestId('audio-player')).toBeInTheDocument();
      }, { timeout: 5000 });
    });
  });

  describe('Community Interaction Flow', () => {
    it('should browse and interact with community tracks', async () => {
      localStorage.setItem('authToken', 'mock-token');

      renderWithProviders(<App />);

      // Navigate to community
      await user.click(screen.getByRole('link', { name: /community/i }));

      await waitFor(() => {
        expect(screen.getByText(/trending tracks/i)).toBeInTheDocument();
      });

      // View trending tracks
      expect(screen.getByText('Summer Vibes')).toBeInTheDocument();
      expect(screen.getByText('Night Drive')).toBeInTheDocument();

      // Play a track
      const firstTrack = screen.getByText('Summer Vibes').closest('article');
      await user.click(within(firstTrack!).getByRole('button', { name: /play/i }));

      await waitFor(() => {
        expect(screen.getByTestId('community-player')).toBeInTheDocument();
      });

      // Like a track
      await user.click(within(firstTrack!).getByRole('button', { name: /like/i }));

      await waitFor(() => {
        expect(within(firstTrack!).getByText('246')).toBeInTheDocument(); // 245 + 1
      });

      // Share track
      await user.click(within(firstTrack!).getByRole('button', { name: /share/i }));

      await waitFor(() => {
        expect(screen.getByText(/link copied/i)).toBeInTheDocument();
      });
    });
  });

  describe('Advanced Features Flow', () => {
    it('should use audio editor for post-processing', async () => {
      localStorage.setItem('authToken', 'mock-token');

      server.use(
        rest.get('/api/projects/:id', (req, res, ctx) => {
          return res(
            ctx.json({
              id: req.params.id,
              name: 'Jazz Project',
              generations: [
                {
                  id: 'gen-1',
                  audioUrl: '/audio/jazz.wav',
                  waveformData: Array(1000).fill(0).map(() => Math.random()),
                },
              ],
            })
          );
        })
      );

      renderWithProviders(<App />);

      // Open project
      await user.click(screen.getByText('Jazz Project'));

      await waitFor(() => {
        expect(screen.getByTestId('project-workspace')).toBeInTheDocument();
      });

      // Open audio editor
      await user.click(screen.getByRole('button', { name: /edit audio/i }));

      await waitFor(() => {
        expect(screen.getByTestId('audio-editor')).toBeInTheDocument();
      });

      // Apply effects
      await user.click(screen.getByRole('button', { name: /effects/i }));
      await user.click(screen.getByText('Reverb'));

      // Adjust parameters
      const reverbControl = screen.getByLabelText(/room size/i);
      await user.clear(reverbControl);
      await user.type(reverbControl, '0.7');

      // Apply effect
      await user.click(screen.getByRole('button', { name: /apply/i }));

      await waitFor(() => {
        expect(screen.getByText(/effect applied/i)).toBeInTheDocument();
      });

      // Export edited audio
      await user.click(screen.getByRole('button', { name: /export/i }));

      await waitFor(() => {
        expect(screen.getByText(/export complete/i)).toBeInTheDocument();
      });
    });

    it('should handle collaborative features', async () => {
      localStorage.setItem('authToken', 'mock-token');

      server.use(
        rest.post('/api/projects/:id/collaborate', async (req, res, ctx) => {
          const { email } = await req.json();
          return res(
            ctx.json({
              success: true,
              collaborator: { email, role: 'editor' },
            })
          );
        }),

        rest.get('/api/projects/:id/activity', (req, res, ctx) => {
          return res(
            ctx.json({
              activities: [
                {
                  id: '1',
                  user: 'User123',
                  action: 'generated new track',
                  timestamp: new Date().toISOString(),
                },
                {
                  id: '2',
                  user: 'User456',
                  action: 'added comment',
                  timestamp: new Date().toISOString(),
                },
              ],
            })
          );
        })
      );

      renderWithProviders(<App />);

      // Open project
      await user.click(screen.getByText('Jazz Project'));

      // Open collaboration panel
      await user.click(screen.getByRole('button', { name: /collaborate/i }));

      await waitFor(() => {
        expect(screen.getByText(/invite collaborators/i)).toBeInTheDocument();
      });

      // Invite collaborator
      await user.type(
        screen.getByPlaceholderText(/email address/i),
        'collaborator@example.com'
      );
      await user.click(screen.getByRole('button', { name: /send invite/i }));

      await waitFor(() => {
        expect(screen.getByText(/invitation sent/i)).toBeInTheDocument();
      });

      // View activity feed
      await user.click(screen.getByRole('tab', { name: /activity/i }));

      await waitFor(() => {
        expect(screen.getByText(/generated new track/i)).toBeInTheDocument();
        expect(screen.getByText(/added comment/i)).toBeInTheDocument();
      });
    });
  });

  describe('Error Recovery Flows', () => {
    it('should handle and recover from generation failures', async () => {
      localStorage.setItem('authToken', 'mock-token');

      let attemptCount = 0;
      server.use(
        rest.post('/api/generate', (req, res, ctx) => {
          attemptCount++;
          if (attemptCount === 1) {
            return res(
              ctx.status(500),
              ctx.json({ error: 'Server overloaded' })
            );
          }
          return res(
            ctx.json({
              id: 'gen-retry',
              audioUrl: '/audio/retry.wav',
            })
          );
        })
      );

      renderWithProviders(<App />);

      // Navigate to generation
      await user.click(screen.getByRole('button', { name: /new generation/i }));

      await user.type(
        screen.getByPlaceholderText(/describe your music/i),
        'Test generation'
      );

      await user.click(screen.getByRole('button', { name: /generate/i }));

      // Wait for error
      await waitFor(() => {
        expect(screen.getByText(/server overloaded/i)).toBeInTheDocument();
      });

      // Retry
      await user.click(screen.getByRole('button', { name: /try again/i }));

      // Should succeed
      await waitFor(() => {
        expect(screen.getByTestId('audio-player')).toBeInTheDocument();
      });
    });

    it('should handle session expiry gracefully', async () => {
      localStorage.setItem('authToken', 'expired-token');

      server.use(
        rest.get('/api/user/profile', (req, res, ctx) => {
          return res(
            ctx.status(401),
            ctx.json({ error: 'Session expired' })
          );
        })
      );

      renderWithProviders(<App />);

      // Should redirect to login
      await waitFor(() => {
        expect(screen.getByPlaceholderText(/email/i)).toBeInTheDocument();
        expect(screen.getByText(/session expired/i)).toBeInTheDocument();
      });

      // Re-login
      await user.type(screen.getByPlaceholderText(/email/i), 'test@example.com');
      await user.type(screen.getByPlaceholderText(/password/i), 'password123');
      await user.click(screen.getByRole('button', { name: /log in/i }));

      // Should restore to dashboard
      await waitFor(() => {
        expect(screen.getByText(/your projects/i)).toBeInTheDocument();
      });
    });
  });

  describe('Performance and Optimization', () => {
    it('should lazy load heavy components', async () => {
      localStorage.setItem('authToken', 'mock-token');

      renderWithProviders(<App />);

      // Initially audio editor should not be loaded
      expect(screen.queryByTestId('audio-editor')).not.toBeInTheDocument();

      // Navigate to editor
      await user.click(screen.getByText('Jazz Project'));
      await user.click(screen.getByRole('button', { name: /edit audio/i }));

      // Should show loading state
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();

      // Editor loads
      await waitFor(() => {
        expect(screen.getByTestId('audio-editor')).toBeInTheDocument();
      });
    });

    it('should cache API responses appropriately', async () => {
      localStorage.setItem('authToken', 'mock-token');

      let projectCallCount = 0;
      server.use(
        rest.get('/api/projects', (req, res, ctx) => {
          projectCallCount++;
          return res(
            ctx.json({
              projects: [
                { id: '1', name: 'Cached Project' },
              ],
            })
          );
        })
      );

      renderWithProviders(<App />);

      // First load
      await waitFor(() => {
        expect(screen.getByText('Cached Project')).toBeInTheDocument();
      });

      expect(projectCallCount).toBe(1);

      // Navigate away and back
      await user.click(screen.getByRole('link', { name: /community/i }));
      await user.click(screen.getByRole('link', { name: /projects/i }));

      // Should use cached data
      expect(screen.getByText('Cached Project')).toBeInTheDocument();
      expect(projectCallCount).toBe(1); // No additional call
    });
  });
});