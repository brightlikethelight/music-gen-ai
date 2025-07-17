/**
 * ErrorBoundary Component Tests
 * 
 * Comprehensive tests for the ErrorBoundary component including:
 * - Error catching and display
 * - Custom fallback rendering
 * - Error logging
 * - Reset functionality
 * - Component lifecycle
 * - Edge cases
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ErrorBoundary } from '../ErrorBoundary';

// Mock console methods
const mockConsoleError = jest.spyOn(console, 'error').mockImplementation();

// Mock ErrorFallback component
jest.mock('../ErrorFallback', () => ({
  ErrorFallback: ({ error, reset }: any) => (
    <div data-testid="error-fallback">
      <p>Error: {error.message}</p>
      <button onClick={reset} data-testid="reset-button">Reset</button>
    </div>
  ),
}));

// Component that throws an error
const ThrowError: React.FC<{ shouldThrow?: boolean; error?: Error }> = ({ 
  shouldThrow = true, 
  error = new Error('Test error') 
}) => {
  if (shouldThrow) {
    throw error;
  }
  return <div data-testid="child-component">Child content</div>;
};

// Component that throws error on click
const ThrowErrorOnClick: React.FC = () => {
  const [shouldThrow, setShouldThrow] = React.useState(false);

  if (shouldThrow) {
    throw new Error('Clicked error');
  }

  return (
    <button onClick={() => setShouldThrow(true)} data-testid="trigger-error">
      Trigger Error
    </button>
  );
};

describe('ErrorBoundary', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    mockConsoleError.mockClear();
  });

  afterAll(() => {
    mockConsoleError.mockRestore();
  });

  describe('Normal Rendering', () => {
    it('should render children when there is no error', () => {
      render(
        <ErrorBoundary>
          <div data-testid="child-component">Child content</div>
        </ErrorBoundary>
      );

      expect(screen.getByTestId('child-component')).toBeInTheDocument();
      expect(screen.getByText('Child content')).toBeInTheDocument();
    });

    it('should render multiple children', () => {
      render(
        <ErrorBoundary>
          <div>Child 1</div>
          <div>Child 2</div>
          <div>Child 3</div>
        </ErrorBoundary>
      );

      expect(screen.getByText('Child 1')).toBeInTheDocument();
      expect(screen.getByText('Child 2')).toBeInTheDocument();
      expect(screen.getByText('Child 3')).toBeInTheDocument();
    });
  });

  describe('Error Catching', () => {
    it('should catch errors and display fallback', () => {
      render(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
      expect(screen.getByText('Error: Test error')).toBeInTheDocument();
    });

    it('should catch errors thrown after initial render', async () => {
      render(
        <ErrorBoundary>
          <ThrowErrorOnClick />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('trigger-error')).toBeInTheDocument();

      await user.click(screen.getByTestId('trigger-error'));

      await waitFor(() => {
        expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
        expect(screen.getByText('Error: Clicked error')).toBeInTheDocument();
      });
    });

    it('should log errors to console', () => {
      const testError = new Error('Console test error');
      
      render(
        <ErrorBoundary>
          <ThrowError error={testError} />
        </ErrorBoundary>
      );

      expect(mockConsoleError).toHaveBeenCalledWith(
        'ErrorBoundary caught error:',
        testError,
        expect.objectContaining({
          componentStack: expect.any(String),
        })
      );
    });

    it('should handle errors with no message', () => {
      const emptyError = new Error();
      
      render(
        <ErrorBoundary>
          <ThrowError error={emptyError} />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
    });
  });

  describe('Custom Fallback', () => {
    it('should render custom fallback when provided', () => {
      const customFallback = (error: Error, reset: () => void) => (
        <div data-testid="custom-fallback">
          <h1>Custom Error</h1>
          <p>{error.message}</p>
          <button onClick={reset} data-testid="custom-reset">
            Custom Reset
          </button>
        </div>
      );

      render(
        <ErrorBoundary fallback={customFallback}>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();
      expect(screen.getByText('Custom Error')).toBeInTheDocument();
      expect(screen.getByText('Test error')).toBeInTheDocument();
      expect(screen.getByTestId('custom-reset')).toBeInTheDocument();
    });

    it('should pass error and reset to custom fallback', async () => {
      let capturedError: Error | null = null;
      let capturedReset: (() => void) | null = null;

      const customFallback = (error: Error, reset: () => void) => {
        capturedError = error;
        capturedReset = reset;
        return <div data-testid="custom-fallback">Error caught</div>;
      };

      render(
        <ErrorBoundary fallback={customFallback}>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(capturedError).toBeInstanceOf(Error);
      expect(capturedError?.message).toBe('Test error');
      expect(typeof capturedReset).toBe('function');
    });
  });

  describe('Error Handler Callback', () => {
    it('should call onError callback when error occurs', () => {
      const mockOnError = jest.fn();
      const testError = new Error('Callback test error');

      render(
        <ErrorBoundary onError={mockOnError}>
          <ThrowError error={testError} />
        </ErrorBoundary>
      );

      expect(mockOnError).toHaveBeenCalledWith(
        testError,
        expect.objectContaining({
          componentStack: expect.any(String),
        })
      );
    });

    it('should call onError before rendering fallback', () => {
      let onErrorCalled = false;
      let fallbackRendered = false;

      const mockOnError = jest.fn(() => {
        onErrorCalled = true;
        expect(fallbackRendered).toBe(false);
      });

      const customFallback = (error: Error) => {
        fallbackRendered = true;
        expect(onErrorCalled).toBe(true);
        return <div>Fallback</div>;
      };

      render(
        <ErrorBoundary onError={mockOnError} fallback={customFallback}>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(mockOnError).toHaveBeenCalled();
    });
  });

  describe('Reset Functionality', () => {
    it('should reset error state and re-render children', async () => {
      const { rerender } = render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();

      await user.click(screen.getByTestId('reset-button'));

      // After reset, should try to render children again
      // Since ThrowError will throw again, it should show error again
      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
    });

    it('should successfully recover after fixing error condition', async () => {
      let shouldThrow = true;

      const ConditionalError: React.FC = () => {
        if (shouldThrow) {
          throw new Error('Conditional error');
        }
        return <div data-testid="success">Success!</div>;
      };

      render(
        <ErrorBoundary>
          <ConditionalError />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();

      // Fix the error condition
      shouldThrow = false;

      // Reset the error boundary
      await user.click(screen.getByTestId('reset-button'));

      // Should now render successfully
      expect(screen.getByTestId('success')).toBeInTheDocument();
      expect(screen.getByText('Success!')).toBeInTheDocument();
    });
  });

  describe('Component Lifecycle', () => {
    it('should update state through getDerivedStateFromError', () => {
      render(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );

      // The fact that we see the error fallback proves getDerivedStateFromError worked
      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
    });

    it('should handle componentDidCatch lifecycle', () => {
      const mockOnError = jest.fn();

      render(
        <ErrorBoundary onError={mockOnError}>
          <ThrowError />
        </ErrorBoundary>
      );

      // componentDidCatch should have been called (evidenced by onError being called)
      expect(mockOnError).toHaveBeenCalled();
      expect(mockConsoleError).toHaveBeenCalled();
    });
  });

  describe('Nested Error Boundaries', () => {
    it('should catch errors in nested error boundaries', () => {
      render(
        <ErrorBoundary>
          <div>
            <ErrorBoundary>
              <ThrowError />
            </ErrorBoundary>
          </div>
        </ErrorBoundary>
      );

      // Inner error boundary should catch the error
      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
    });

    it('should propagate errors when inner boundary throws', () => {
      const ThrowingFallback = () => {
        throw new Error('Fallback error');
      };

      render(
        <ErrorBoundary>
          <ErrorBoundary fallback={() => <ThrowingFallback />}>
            <ThrowError />
          </ErrorBoundary>
        </ErrorBoundary>
      );

      // Outer error boundary should catch the error from inner fallback
      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle null/undefined errors gracefully', () => {
      // @ts-ignore - Testing edge case
      const NullError: React.FC = () => {
        throw null;
      };

      render(
        <ErrorBoundary>
          <NullError />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
    });

    it('should handle non-Error objects', () => {
      const StringError: React.FC = () => {
        // eslint-disable-next-line no-throw-literal
        throw 'String error';
      };

      render(
        <ErrorBoundary>
          <StringError />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
    });

    it('should handle errors during async operations', async () => {
      const AsyncError: React.FC = () => {
        React.useEffect(() => {
          // This will be caught by the global error handler, not ErrorBoundary
          setTimeout(() => {
            throw new Error('Async error');
          }, 0);
        }, []);

        return <div>Async component</div>;
      };

      render(
        <ErrorBoundary>
          <AsyncError />
        </ErrorBoundary>
      );

      // Async errors in setTimeout won't be caught by ErrorBoundary
      expect(screen.getByText('Async component')).toBeInTheDocument();
    });

    it('should handle errors in event handlers with error boundaries', async () => {
      const EventError: React.FC = () => {
        const [hasError, setHasError] = React.useState(false);

        if (hasError) {
          throw new Error('Event handler error');
        }

        return (
          <button onClick={() => setHasError(true)} data-testid="event-button">
            Click me
          </button>
        );
      };

      render(
        <ErrorBoundary>
          <EventError />
        </ErrorBoundary>
      );

      await user.click(screen.getByTestId('event-button'));

      await waitFor(() => {
        expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
      });
    });

    it('should preserve error boundary state across re-renders', () => {
      const { rerender } = render(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();

      // Re-render with same props
      rerender(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );

      // Should still show error fallback
      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
    });
  });

  describe('Memory Management', () => {
    it('should clean up state on unmount', () => {
      const { unmount } = render(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-fallback')).toBeInTheDocument();

      unmount();

      // Component should unmount cleanly without errors
      expect(mockConsoleError).toHaveBeenCalledWith(
        'ErrorBoundary caught error:',
        expect.any(Error),
        expect.any(Object)
      );
    });
  });
});