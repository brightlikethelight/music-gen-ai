/**
 * NetworkErrorBoundary Component Tests
 * 
 * Comprehensive tests for the NetworkErrorBoundary component including:
 * - Online/offline detection
 * - Network error handling
 * - Retry functionality
 * - Custom fallback
 * - Browser events
 * - Exponential backoff
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { NetworkErrorBoundary } from '../NetworkErrorBoundary';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Save original navigator.onLine
const originalNavigatorOnLine = Object.getOwnPropertyDescriptor(window.navigator, 'onLine');

// Helper to mock navigator.onLine
const mockNavigatorOnLine = (isOnline: boolean) => {
  Object.defineProperty(window.navigator, 'onLine', {
    writable: true,
    configurable: true,
    value: isOnline,
  });
};

// Helper to trigger browser events
const triggerEvent = (eventName: string, detail?: any) => {
  const event = detail ? new CustomEvent(eventName, { detail }) : new Event(eventName);
  window.dispatchEvent(event);
};

describe('NetworkErrorBoundary', () => {
  const user = userEvent.setup();
  const mockOnRetry = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    mockNavigatorOnLine(true);
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
    // Restore original navigator.onLine
    if (originalNavigatorOnLine) {
      Object.defineProperty(window.navigator, 'onLine', originalNavigatorOnLine);
    }
  });

  describe('Normal Rendering', () => {
    it('should render children when online', () => {
      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByTestId('child-content')).toBeInTheDocument();
      expect(screen.getByText('App Content')).toBeInTheDocument();
    });

    it('should render multiple children when online', () => {
      renderWithProviders(
        <NetworkErrorBoundary>
          <div>Child 1</div>
          <div>Child 2</div>
          <div>Child 3</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByText('Child 1')).toBeInTheDocument();
      expect(screen.getByText('Child 2')).toBeInTheDocument();
      expect(screen.getByText('Child 3')).toBeInTheDocument();
    });
  });

  describe('Offline Detection', () => {
    it('should show offline UI when navigator.onLine is false', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByText(/no internet connection/i)).toBeInTheDocument();
      expect(screen.getByText(/please check your internet connection/i)).toBeInTheDocument();
      expect(screen.queryByTestId('child-content')).not.toBeInTheDocument();
    });

    it('should show offline UI when offline event is triggered', () => {
      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByTestId('child-content')).toBeInTheDocument();

      act(() => {
        triggerEvent('offline');
      });

      expect(screen.getByText(/no internet connection/i)).toBeInTheDocument();
      expect(screen.queryByTestId('child-content')).not.toBeInTheDocument();
    });

    it('should recover when online event is triggered', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByText(/no internet connection/i)).toBeInTheDocument();

      act(() => {
        mockNavigatorOnLine(true);
        triggerEvent('online');
      });

      expect(screen.getByTestId('child-content')).toBeInTheDocument();
      expect(screen.queryByText(/no internet connection/i)).not.toBeInTheDocument();
    });
  });

  describe('Network Error Detection', () => {
    it('should show network error UI for fetch errors', () => {
      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      const fetchError = new TypeError('Failed to fetch');
      const rejectionEvent = new PromiseRejectionEvent('unhandledrejection', {
        promise: Promise.reject(fetchError),
        reason: fetchError,
      });

      act(() => {
        window.dispatchEvent(rejectionEvent);
      });

      expect(screen.getByText(/network error/i)).toBeInTheDocument();
      expect(screen.getByText(/trouble connecting to our servers/i)).toBeInTheDocument();
    });

    it('should prevent unhandled rejection warning for fetch errors', () => {
      const preventDefaultSpy = jest.fn();
      
      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      const fetchError = new TypeError('Failed to fetch');
      const rejectionEvent = new PromiseRejectionEvent('unhandledrejection', {
        promise: Promise.reject(fetchError),
        reason: fetchError,
      });
      
      // Add preventDefault spy
      Object.defineProperty(rejectionEvent, 'preventDefault', {
        value: preventDefaultSpy,
        writable: true,
      });

      act(() => {
        window.dispatchEvent(rejectionEvent);
      });

      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it('should not handle non-fetch errors', () => {
      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      const otherError = new Error('Some other error');
      const rejectionEvent = new PromiseRejectionEvent('unhandledrejection', {
        promise: Promise.reject(otherError),
        reason: otherError,
      });

      act(() => {
        window.dispatchEvent(rejectionEvent);
      });

      // Should still show child content
      expect(screen.getByTestId('child-content')).toBeInTheDocument();
    });
  });

  describe('Retry Functionality', () => {
    it('should handle retry when online', async () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary onRetry={mockOnRetry}>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      mockNavigatorOnLine(true);

      await user.click(screen.getByRole('button', { name: /try again/i }));

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      await waitFor(() => {
        expect(mockOnRetry).toHaveBeenCalled();
        expect(screen.getByTestId('child-content')).toBeInTheDocument();
      });
    });

    it('should show loading state during retry', async () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      await user.click(screen.getByRole('button', { name: /try again/i }));

      expect(screen.getByText(/checking connection/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /checking connection/i })).toBeDisabled();

      act(() => {
        jest.advanceTimersByTime(1000);
      });
    });

    it('should increment retry count', async () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      // First retry
      await user.click(screen.getByRole('button', { name: /try again/i }));
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      await waitFor(() => {
        expect(screen.getByText(/retry attempt 1/i)).toBeInTheDocument();
      });

      // Second retry
      await user.click(screen.getByRole('button', { name: /try again/i }));
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      await waitFor(() => {
        expect(screen.getByText(/retry attempt 2/i)).toBeInTheDocument();
      });
    });

    it('should show exponential backoff delay', async () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      // First retry - 5 seconds
      await user.click(screen.getByRole('button', { name: /try again/i }));
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      await waitFor(() => {
        expect(screen.getByText(/next retry in 5 seconds/i)).toBeInTheDocument();
      });

      // Second retry - 10 seconds
      await user.click(screen.getByRole('button', { name: /try again/i }));
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      await waitFor(() => {
        expect(screen.getByText(/next retry in 10 seconds/i)).toBeInTheDocument();
      });
    });

    it('should cap retry delay at 60 seconds', async () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      // Simulate many retries to exceed max delay
      for (let i = 0; i < 10; i++) {
        await user.click(screen.getByRole('button', { name: /try again/i }));
        act(() => {
          jest.advanceTimersByTime(1000);
        });
      }

      await waitFor(() => {
        expect(screen.getByText(/next retry in 60 seconds/i)).toBeInTheDocument();
      });
    });
  });

  describe('Custom Fallback', () => {
    it('should render custom fallback when provided', () => {
      mockNavigatorOnLine(false);

      const customFallback = <div data-testid="custom-fallback">Custom Error UI</div>;

      renderWithProviders(
        <NetworkErrorBoundary fallback={customFallback}>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();
      expect(screen.getByText('Custom Error UI')).toBeInTheDocument();
      expect(screen.queryByText(/no internet connection/i)).not.toBeInTheDocument();
    });

    it('should use custom fallback for network errors too', () => {
      const customFallback = <div data-testid="custom-fallback">Custom Network Error</div>;

      renderWithProviders(
        <NetworkErrorBoundary fallback={customFallback}>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      const fetchError = new TypeError('Failed to fetch');
      const rejectionEvent = new PromiseRejectionEvent('unhandledrejection', {
        promise: Promise.reject(fetchError),
        reason: fetchError,
      });

      act(() => {
        window.dispatchEvent(rejectionEvent);
      });

      expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();
    });
  });

  describe('UI Elements', () => {
    it('should show refresh page button', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByRole('button', { name: /refresh page/i })).toBeInTheDocument();
    });

    it('should reload page when refresh button clicked', async () => {
      const reloadSpy = jest.spyOn(window.location, 'reload').mockImplementation();
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      await user.click(screen.getByRole('button', { name: /refresh page/i }));

      expect(reloadSpy).toHaveBeenCalled();
      reloadSpy.mockRestore();
    });

    it('should show troubleshooting tips', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByText(/troubleshooting tips/i)).toBeInTheDocument();
      expect(screen.getByText(/check your wi-fi/i)).toBeInTheDocument();
      expect(screen.getByText(/disable vpn/i)).toBeInTheDocument();
      expect(screen.getByText(/clear your browser cache/i)).toBeInTheDocument();
    });

    it('should show browser online/offline status', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByText(/browser offline/i)).toBeInTheDocument();

      act(() => {
        mockNavigatorOnLine(true);
        triggerEvent('online');
      });

      expect(screen.getByText(/browser online/i)).toBeInTheDocument();
    });

    it('should show correct icons', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      // Should show WiFi and slash icons for offline
      const icons = screen.getAllByRole('img', { hidden: true });
      expect(icons.length).toBeGreaterThan(0);
    });
  });

  describe('Event Cleanup', () => {
    it('should clean up event listeners on unmount', () => {
      const removeEventListenerSpy = jest.spyOn(window, 'removeEventListener');

      const { unmount } = renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('online', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('offline', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('unhandledrejection', expect.any(Function));

      removeEventListenerSpy.mockRestore();
    });
  });

  describe('Accessibility', () => {
    it('should have proper headings', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByRole('heading', { name: /no internet connection/i })).toBeInTheDocument();
    });

    it('should be keyboard navigable', async () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      await user.tab();
      expect(screen.getByRole('button', { name: /try again/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /refresh page/i })).toHaveFocus();
    });

    it('should handle keyboard activation', async () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary onRetry={mockOnRetry}>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      mockNavigatorOnLine(true);

      const tryAgainButton = screen.getByRole('button', { name: /try again/i });
      tryAgainButton.focus();
      
      await user.keyboard('{Enter}');

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      await waitFor(() => {
        expect(mockOnRetry).toHaveBeenCalled();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle rapid online/offline changes', () => {
      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      // Rapidly toggle online/offline
      act(() => {
        triggerEvent('offline');
        triggerEvent('online');
        triggerEvent('offline');
        triggerEvent('online');
      });

      // Should end up online
      expect(screen.getByTestId('child-content')).toBeInTheDocument();
    });

    it('should handle multiple network errors', () => {
      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      const fetchError1 = new TypeError('Failed to fetch');
      const fetchError2 = new TypeError('Failed to fetch');

      act(() => {
        window.dispatchEvent(new PromiseRejectionEvent('unhandledrejection', {
          promise: Promise.reject(fetchError1),
          reason: fetchError1,
        }));
        window.dispatchEvent(new PromiseRejectionEvent('unhandledrejection', {
          promise: Promise.reject(fetchError2),
          reason: fetchError2,
        }));
      });

      // Should show network error UI once
      expect(screen.getByText(/network error/i)).toBeInTheDocument();
    });

    it('should handle retry when still offline', async () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      await user.click(screen.getByRole('button', { name: /try again/i }));

      act(() => {
        jest.advanceTimersByTime(1000);
      });

      // Should still show offline UI
      expect(screen.getByText(/no internet connection/i)).toBeInTheDocument();
    });

    it('should reset retry count when going online', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      // Do some retries
      act(async () => {
        await user.click(screen.getByRole('button', { name: /try again/i }));
        jest.advanceTimersByTime(1000);
      });

      expect(screen.getByText(/retry attempt 1/i)).toBeInTheDocument();

      // Go online
      act(() => {
        mockNavigatorOnLine(true);
        triggerEvent('online');
      });

      // Go offline again
      act(() => {
        mockNavigatorOnLine(false);
        triggerEvent('offline');
      });

      // Retry count should be reset
      expect(screen.queryByText(/retry attempt/i)).not.toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('should handle mobile viewport', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      expect(screen.getByText(/no internet connection/i)).toBeInTheDocument();
    });

    it('should constrain content width', () => {
      mockNavigatorOnLine(false);

      renderWithProviders(
        <NetworkErrorBoundary>
          <div data-testid="child-content">App Content</div>
        </NetworkErrorBoundary>
      );

      const container = screen.getByText(/no internet connection/i).closest('.max-w-md');
      expect(container).toBeInTheDocument();
    });
  });
});