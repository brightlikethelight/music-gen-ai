/**
 * Test setup and configuration for React Testing Library
 */

import '@testing-library/jest-dom';
import { configure } from '@testing-library/react';
import { server } from './test-utils/server';

// Configure React Testing Library
configure({
  testIdAttribute: 'data-testid',
  asyncUtilTimeout: 10000, // 10 seconds for async operations
  getElementError: (message, container) => {
    const prettierMessage = [
      message,
      'Here is the HTML structure:',
      container ? container.innerHTML : 'No container provided',
    ].join('\n\n');
    return new Error(prettierMessage);
  },
});

// Global test setup
beforeAll(() => {
  // Start MSW server for API mocking
  server.listen();
  
  // Mock window.matchMedia for responsive design tests
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation((query) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(), // deprecated
      removeListener: jest.fn(), // deprecated
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    })),
  });

  // Mock ResizeObserver for responsive components
  global.ResizeObserver = jest.fn().mockImplementation(() => ({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  }));

  // Mock IntersectionObserver for lazy loading components
  global.IntersectionObserver = jest.fn().mockImplementation(() => ({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  }));

  // Mock Web Audio API for audio components
  const mockAudioContext = {
    createGain: jest.fn(() => ({
      connect: jest.fn(),
      disconnect: jest.fn(),
      gain: { value: 1 },
    })),
    createOscillator: jest.fn(() => ({
      connect: jest.fn(),
      disconnect: jest.fn(),
      start: jest.fn(),
      stop: jest.fn(),
      frequency: { value: 440 },
    })),
    createAnalyser: jest.fn(() => ({
      connect: jest.fn(),
      disconnect: jest.fn(),
      getByteFrequencyData: jest.fn(),
      fftSize: 2048,
    })),
    destination: {},
    sampleRate: 44100,
    currentTime: 0,
    resume: jest.fn(),
    suspend: jest.fn(),
    close: jest.fn(),
    state: 'running',
  };

  global.AudioContext = jest.fn(() => mockAudioContext);
  global.webkitAudioContext = jest.fn(() => mockAudioContext);

  // Mock URL.createObjectURL for file handling
  global.URL.createObjectURL = jest.fn(() => 'mocked-object-url');
  global.URL.revokeObjectURL = jest.fn();

  // Mock localStorage
  const localStorageMock = {
    getItem: jest.fn(),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
  };
  Object.defineProperty(window, 'localStorage', {
    value: localStorageMock,
  });

  // Mock sessionStorage
  const sessionStorageMock = {
    getItem: jest.fn(),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
  };
  Object.defineProperty(window, 'sessionStorage', {
    value: sessionStorageMock,
  });

  // Mock WebSocket for streaming tests
  global.WebSocket = jest.fn().mockImplementation(() => ({
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    send: jest.fn(),
    close: jest.fn(),
    readyState: 1, // OPEN
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
  }));

  // Mock crypto for secure operations
  Object.defineProperty(global, 'crypto', {
    value: {
      randomUUID: jest.fn(() => 'mocked-uuid'),
      getRandomValues: jest.fn((arr) => arr.map(() => Math.floor(Math.random() * 256))),
    },
  });

  // Suppress console errors for known issues during testing
  const originalConsoleError = console.error;
  console.error = (...args) => {
    // Suppress known warnings that don't affect test results
    const message = args[0];
    if (
      typeof message === 'string' &&
      (message.includes('Warning: ReactDOM.render is deprecated') ||
       message.includes('Warning: validateDOMNesting') ||
       message.includes('act(...) warning'))
    ) {
      return;
    }
    originalConsoleError(...args);
  };
});

afterEach(() => {
  // Reset all mocks after each test
  jest.clearAllMocks();
  server.resetHandlers();
  
  // Clear localStorage and sessionStorage
  localStorage.clear();
  sessionStorage.clear();
});

afterAll(() => {
  // Clean up MSW server
  server.close();
  
  // Restore console.error
  console.error = console.error;
});

// Custom matchers for testing
expect.extend({
  toBeAccessible(received) {
    // Simple accessibility check - in real app would use @testing-library/jest-dom
    const hasAriaLabel = received.getAttribute('aria-label');
    const hasRole = received.getAttribute('role');
    const hasTabIndex = received.getAttribute('tabindex');
    
    const pass = !!(hasAriaLabel || hasRole || hasTabIndex);
    
    if (pass) {
      return {
        message: () => `expected ${received} not to be accessible`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be accessible (have aria-label, role, or tabindex)`,
        pass: false,
      };
    }
  },
});

// Global test timeout
jest.setTimeout(30000); // 30 seconds for complex component tests