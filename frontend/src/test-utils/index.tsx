/**
 * Test utilities for React Testing Library
 * 
 * Provides custom render functions, mock providers, and testing helpers
 * for comprehensive component testing with all necessary context.
 */

import React, { ReactElement, ReactNode } from 'react';
import { render, RenderOptions, RenderResult } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from 'styled-components';
import { AuthProvider } from '../contexts/AuthContext';
import { WebSocketProvider } from '../contexts/WebSocketContext';

// Mock theme for styled-components
const mockTheme = {
  colors: {
    primary: '#3B82F6',
    secondary: '#6B7280',
    success: '#10B981',
    warning: '#F59E0B',
    error: '#EF4444',
    background: '#FFFFFF',
    surface: '#F9FAFB',
    text: {
      primary: '#111827',
      secondary: '#6B7280',
      disabled: '#9CA3AF',
    },
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
  },
  borderRadius: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
  },
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
  },
};

// Mock user for authentication tests
export const mockUser = {
  id: 'user-123',
  email: 'test@example.com',
  username: 'testuser',
  roles: ['user'],
  tier: 'free',
  isVerified: true,
  createdAt: new Date(),
  lastLogin: new Date(),
};

export const mockAdminUser = {
  ...mockUser,
  id: 'admin-123',
  email: 'admin@example.com',
  username: 'admin',
  roles: ['admin', 'user'],
  tier: 'enterprise',
};

// Mock authentication context values
export const mockAuthContextValue = {
  user: mockUser,
  isAuthenticated: true,
  isLoading: false,
  login: jest.fn(),
  logout: jest.fn(),
  register: jest.fn(),
  refreshToken: jest.fn(),
  updateProfile: jest.fn(),
};

export const mockUnauthenticatedContextValue = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  login: jest.fn(),
  logout: jest.fn(),
  register: jest.fn(),
  refreshToken: jest.fn(),
  updateProfile: jest.fn(),
};

// Mock WebSocket context values
export const mockWebSocketContextValue = {
  socket: null,
  isConnected: false,
  connect: jest.fn(),
  disconnect: jest.fn(),
  send: jest.fn(),
  subscribe: jest.fn(),
  unsubscribe: jest.fn(),
};

// Custom providers wrapper
interface ProvidersProps {
  children: ReactNode;
  initialRoute?: string;
  authValue?: any;
  wsValue?: any;
  queryClient?: QueryClient;
}

const AllTheProviders: React.FC<ProvidersProps> = ({
  children,
  initialRoute = '/',
  authValue = mockAuthContextValue,
  wsValue = mockWebSocketContextValue,
  queryClient,
}) => {
  // Create a fresh QueryClient for each test to avoid state leakage
  const testQueryClient = queryClient || new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0, // Disable caching
      },
      mutations: {
        retry: false,
      },
    },
  });

  // Set initial route for router
  if (initialRoute !== '/') {
    window.history.pushState({}, 'Test page', initialRoute);
  }

  return (
    <BrowserRouter>
      <QueryClientProvider client={testQueryClient}>
        <ThemeProvider theme={mockTheme}>
          <AuthProvider value={authValue}>
            <WebSocketProvider value={wsValue}>
              {children}
            </WebSocketProvider>
          </AuthProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </BrowserRouter>
  );
};

// Custom render function with all providers
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialRoute?: string;
  authValue?: any;
  wsValue?: any;
  queryClient?: QueryClient;
}

export const renderWithProviders = (
  ui: ReactElement,
  options: CustomRenderOptions = {}
): RenderResult => {
  const {
    initialRoute,
    authValue,
    wsValue,
    queryClient,
    ...renderOptions
  } = options;

  const Wrapper: React.FC<{ children: ReactNode }> = ({ children }) => (
    <AllTheProviders
      initialRoute={initialRoute}
      authValue={authValue}
      wsValue={wsValue}
      queryClient={queryClient}
    >
      {children}
    </AllTheProviders>
  );

  return render(ui, { wrapper: Wrapper, ...renderOptions });
};

// Render with unauthenticated user
export const renderWithUnauthenticatedUser = (
  ui: ReactElement,
  options: CustomRenderOptions = {}
) => {
  return renderWithProviders(ui, {
    ...options,
    authValue: mockUnauthenticatedContextValue,
  });
};

// Render with admin user
export const renderWithAdminUser = (
  ui: ReactElement,
  options: CustomRenderOptions = {}
) => {
  return renderWithProviders(ui, {
    ...options,
    authValue: {
      ...mockAuthContextValue,
      user: mockAdminUser,
    },
  });
};

// Render with loading state
export const renderWithLoadingAuth = (
  ui: ReactElement,
  options: CustomRenderOptions = {}
) => {
  return renderWithProviders(ui, {
    ...options,
    authValue: {
      ...mockUnauthenticatedContextValue,
      isLoading: true,
    },
  });
};

// Mock API responses
export const mockApiResponses = {
  // Authentication responses
  loginSuccess: {
    success: true,
    user: mockUser,
    csrfToken: 'mock-csrf-token',
  },
  loginError: {
    detail: 'Invalid credentials',
  },
  registerSuccess: {
    success: true,
    user: mockUser,
    csrfToken: 'mock-csrf-token',
    message: 'Registration successful',
  },

  // Generation responses
  generationRequest: {
    task_id: 'task-123',
    status: 'pending',
  },
  generationStatus: {
    task_id: 'task-123',
    status: 'completed',
    audio_url: '/download/task-123',
    duration: 30.0,
    metadata: {
      prompt: 'upbeat electronic music',
      model: 'musicgen-small',
    },
  },
  generationError: {
    task_id: 'task-123',
    status: 'failed',
    error: 'Generation failed due to model error',
  },

  // Project responses
  projects: [
    {
      id: 'project-1',
      name: 'My First Project',
      description: 'A test project',
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      tracks: [],
    },
    {
      id: 'project-2',
      name: 'Another Project',
      description: 'Another test project',
      created_at: '2024-01-02T00:00:00Z',
      updated_at: '2024-01-02T00:00:00Z',
      tracks: [],
    },
  ],
  createProject: {
    id: 'project-new',
    name: 'New Project',
    description: 'A newly created project',
    created_at: '2024-01-03T00:00:00Z',
    updated_at: '2024-01-03T00:00:00Z',
    tracks: [],
  },
};

// User interaction helpers
export const userInteractionHelpers = {
  // Fill form helper
  fillLoginForm: async (getByLabelText: any, email = 'test@example.com', password = 'password123') => {
    const { fireEvent } = await import('@testing-library/react');
    const emailInput = getByLabelText(/email/i);
    const passwordInput = getByLabelText(/password/i);
    
    fireEvent.change(emailInput, { target: { value: email } });
    fireEvent.change(passwordInput, { target: { value: password } });
  },

  // Fill registration form helper
  fillRegistrationForm: async (getByLabelText: any, userData = {
    email: 'test@example.com',
    username: 'testuser',
    password: 'password123',
  }) => {
    const { fireEvent } = await import('@testing-library/react');
    const emailInput = getByLabelText(/email/i);
    const usernameInput = getByLabelText(/username/i);
    const passwordInput = getByLabelText(/password/i);
    
    fireEvent.change(emailInput, { target: { value: userData.email } });
    fireEvent.change(usernameInput, { target: { value: userData.username } });
    fireEvent.change(passwordInput, { target: { value: userData.password } });
  },

  // Submit form helper
  submitForm: async (getByRole: any, buttonText = /submit|login|register|create/i) => {
    const { fireEvent } = await import('@testing-library/react');
    const submitButton = getByRole('button', { name: buttonText });
    fireEvent.click(submitButton);
  },

  // File upload helper
  uploadFile: async (input: HTMLElement, file: File) => {
    const { fireEvent } = await import('@testing-library/react');
    fireEvent.change(input, { target: { files: [file] } });
  },

  // Drag and drop helper
  dragAndDrop: async (source: HTMLElement, target: HTMLElement) => {
    const { fireEvent } = await import('@testing-library/react');
    
    fireEvent.dragStart(source);
    fireEvent.dragEnter(target);
    fireEvent.dragOver(target);
    fireEvent.drop(target);
    fireEvent.dragEnd(source);
  },
};

// Accessibility testing helpers
export const accessibilityHelpers = {
  // Check for proper heading hierarchy
  checkHeadingHierarchy: (container: HTMLElement) => {
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6');
    let previousLevel = 0;
    
    for (const heading of headings) {
      const currentLevel = parseInt(heading.tagName[1]);
      if (currentLevel > previousLevel + 1) {
        throw new Error(`Heading hierarchy violation: found h${currentLevel} after h${previousLevel}`);
      }
      previousLevel = currentLevel;
    }
  },

  // Check for form labels
  checkFormLabels: (container: HTMLElement) => {
    const inputs = container.querySelectorAll('input, select, textarea');
    
    for (const input of inputs) {
      const hasLabel = input.getAttribute('aria-label') || 
                      input.getAttribute('aria-labelledby') ||
                      container.querySelector(`label[for="${input.id}"]`);
      
      if (!hasLabel) {
        throw new Error(`Form input without label: ${input.outerHTML}`);
      }
    }
  },

  // Check for button accessibility
  checkButtonAccessibility: (container: HTMLElement) => {
    const buttons = container.querySelectorAll('button, [role="button"]');
    
    for (const button of buttons) {
      const hasAccessibleName = button.textContent?.trim() ||
                               button.getAttribute('aria-label') ||
                               button.getAttribute('aria-labelledby');
      
      if (!hasAccessibleName) {
        throw new Error(`Button without accessible name: ${button.outerHTML}`);
      }
    }
  },
};

// Responsive design testing helpers
export const responsiveHelpers = {
  // Set viewport size
  setViewportSize: (width: number, height: number) => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: width,
    });
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: height,
    });
    window.dispatchEvent(new Event('resize'));
  },

  // Common viewport sizes
  viewports: {
    mobile: { width: 375, height: 667 },
    tablet: { width: 768, height: 1024 },
    desktop: { width: 1024, height: 768 },
    wide: { width: 1920, height: 1080 },
  },

  // Test responsive behavior
  testResponsiveComponent: async (component: ReactElement, testFn: (size: string) => void) => {
    for (const [sizeName, { width, height }] of Object.entries(responsiveHelpers.viewports)) {
      responsiveHelpers.setViewportSize(width, height);
      await testFn(sizeName);
    }
  },
};

// Performance testing helpers
export const performanceHelpers = {
  // Measure render time
  measureRenderTime: async (renderFn: () => void): Promise<number> => {
    const start = performance.now();
    renderFn();
    const end = performance.now();
    return end - start;
  },

  // Check for memory leaks
  checkMemoryLeaks: (initialMemory: number, threshold = 10): boolean => {
    const currentMemory = (performance as any).memory?.usedJSHeapSize || 0;
    const difference = currentMemory - initialMemory;
    return difference < threshold * 1024 * 1024; // 10MB threshold
  },
};

// Re-export everything from React Testing Library for convenience
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';