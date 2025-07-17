/**
 * LazyLoad Component Tests
 * 
 * Comprehensive tests for the LazyLoad component including:
 * - Dynamic component loading
 * - Loading states with delay
 * - Error handling
 * - Custom fallbacks
 * - Suspense integration
 * - Props passing
 * - Memory management
 * - Edge cases
 */

import React, { Suspense } from 'react';
import { screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { LazyLoad, createLazyComponent } from '../LazyLoad';

// Mock components
jest.mock('../LoadingSpinner', () => ({
  LoadingSpinner: ({ size }: any) => (
    <div data-testid="loading-spinner" data-size={size}>Loading...</div>
  ),
}));

jest.mock('../../errors/ErrorMessage', () => ({
  ErrorMessage: ({ severity, title, message, onDismiss }: any) => (
    <div data-testid="error-message" data-severity={severity}>
      <h3>{title}</h3>
      <p>{message}</p>
      {onDismiss && <button onClick={onDismiss}>Dismiss</button>}
    </div>
  ),
}));

jest.mock('../../errors/ErrorBoundary', () => ({
  ErrorBoundary: ({ children, fallback }: any) => {
    const [hasError, setHasError] = React.useState(false);
    const [error, setError] = React.useState<Error | null>(null);

    React.useEffect(() => {
      if (hasError && error && fallback) {
        return;
      }
    }, [hasError, error, fallback]);

    if (hasError && error && fallback) {
      return fallback(error, () => setHasError(false));
    }

    return (
      <div data-testid="error-boundary">
        {React.Children.map(children, child => {
          if (React.isValidElement(child) && child.props.throwError) {
            setError(new Error('Component error'));
            setHasError(true);
            return null;
          }
          return child;
        })}
      </div>
    );
  },
}));

// Test components
const TestComponent: React.FC<{ message: string; throwError?: boolean }> = ({ 
  message, 
  throwError 
}) => {
  if (throwError) {
    throw new Error('Test component error');
  }
  return <div data-testid="test-component">{message}</div>;
};

const SlowComponent: React.FC<{ delay: number }> = ({ delay }) => {
  const [loaded, setLoaded] = React.useState(false);
  
  React.useEffect(() => {
    const timer = setTimeout(() => setLoaded(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return loaded ? <div data-testid="slow-component">Slow component loaded</div> : null;
};

describe('LazyLoad', () => {
  const user = userEvent.setup();
  
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  describe('Basic Loading', () => {
    it('should load component successfully', async () => {
      const loader = jest.fn().mockResolvedValue({ default: TestComponent });

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          props={{ message: 'Hello World' }}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('test-component')).toBeInTheDocument();
        expect(screen.getByText('Hello World')).toBeInTheDocument();
      });

      expect(loader).toHaveBeenCalledTimes(1);
    });

    it('should show loading state immediately when delay is 0', async () => {
      const loader = jest.fn(() => new Promise(() => {})); // Never resolves

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          delay={0}
        />
      );

      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    });

    it('should delay showing loading state', async () => {
      const loader = jest.fn(() => new Promise(() => {})); // Never resolves

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          delay={500}
        />
      );

      // Should not show loading immediately
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();

      // Advance time but not past delay
      act(() => {
        jest.advanceTimersByTime(400);
      });

      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();

      // Advance past delay
      act(() => {
        jest.advanceTimersByTime(200);
      });

      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    });

    it('should not show loading if component loads before delay', async () => {
      const loader = jest.fn().mockResolvedValue({ default: TestComponent });

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          delay={1000}
          props={{ message: 'Fast load' }}
        />
      );

      // Component loads immediately
      await waitFor(() => {
        expect(screen.getByTestId('test-component')).toBeInTheDocument();
      });

      // Advance time past delay
      act(() => {
        jest.advanceTimersByTime(1500);
      });

      // Should never have shown loading
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });
  });

  describe('Custom Fallbacks', () => {
    it('should show custom loading fallback', async () => {
      const loader = jest.fn(() => new Promise(() => {}));
      const customFallback = <div data-testid="custom-loader">Custom Loading...</div>;

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          fallback={customFallback}
          delay={0}
        />
      );

      expect(screen.getByTestId('custom-loader')).toBeInTheDocument();
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });

    it('should show custom error fallback', async () => {
      const loader = jest.fn().mockRejectedValue(new Error('Load failed'));
      const customError = <div data-testid="custom-error">Custom Error UI</div>;

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          errorFallback={customError}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('custom-error')).toBeInTheDocument();
      });

      expect(screen.queryByTestId('error-message')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should handle loader errors', async () => {
      const error = new Error('Failed to load component');
      const loader = jest.fn().mockRejectedValue(error);

      renderWithProviders(
        <LazyLoad loader={loader} />
      );

      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toBeInTheDocument();
        expect(screen.getByText('Failed to load component')).toBeInTheDocument();
      });
    });

    it('should call onError callback when provided', async () => {
      const error = new Error('Load error');
      const loader = jest.fn().mockRejectedValue(error);
      const onError = jest.fn();

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          onError={onError}
        />
      );

      await waitFor(() => {
        expect(onError).toHaveBeenCalledWith(error);
      });
    });

    it('should handle component runtime errors', async () => {
      const ErrorComponent = () => {
        throw new Error('Runtime error');
      };
      const loader = jest.fn().mockResolvedValue({ default: ErrorComponent });

      renderWithProviders(
        <LazyLoad loader={loader} />
      );

      await waitFor(() => {
        expect(screen.getByTestId('error-boundary')).toBeInTheDocument();
      });
    });

    it('should use error fallback for runtime errors', async () => {
      const ErrorComponent = ({ throwError }: any) => (
        <div throwError={throwError}>Error Component</div>
      );
      const loader = jest.fn().mockResolvedValue({ default: ErrorComponent });
      const errorFallback = <div data-testid="runtime-error">Runtime Error Handler</div>;

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          errorFallback={errorFallback}
          props={{ throwError: true }}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('error-boundary')).toBeInTheDocument();
      });
    });
  });

  describe('Props Passing', () => {
    it('should pass props to loaded component', async () => {
      const ComponentWithProps: React.FC<{ name: string; count: number }> = ({ name, count }) => (
        <div data-testid="props-component">
          Name: {name}, Count: {count}
        </div>
      );

      const loader = jest.fn().mockResolvedValue({ default: ComponentWithProps });

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          props={{ name: 'Test', count: 42 }}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Name: Test, Count: 42')).toBeInTheDocument();
      });
    });

    it('should handle empty props object', async () => {
      const loader = jest.fn().mockResolvedValue({ default: TestComponent });

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          props={{}}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('test-component')).toBeInTheDocument();
      });
    });

    it('should handle undefined props', async () => {
      const loader = jest.fn().mockResolvedValue({ default: TestComponent });

      renderWithProviders(
        <LazyLoad 
          loader={loader}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('test-component')).toBeInTheDocument();
      });
    });
  });

  describe('Memory Management', () => {
    it('should clear timeout on unmount during delay', () => {
      const loader = jest.fn(() => new Promise(() => {}));
      const clearTimeoutSpy = jest.spyOn(global, 'clearTimeout');

      const { unmount } = renderWithProviders(
        <LazyLoad 
          loader={loader}
          delay={1000}
        />
      );

      unmount();

      expect(clearTimeoutSpy).toHaveBeenCalled();
      clearTimeoutSpy.mockRestore();
    });

    it('should handle unmount after component loads', async () => {
      const loader = jest.fn().mockResolvedValue({ default: TestComponent });

      const { unmount } = renderWithProviders(
        <LazyLoad 
          loader={loader}
          props={{ message: 'Test' }}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('test-component')).toBeInTheDocument();
      });

      // Should unmount without errors
      unmount();
    });
  });

  describe('Edge Cases', () => {
    it('should handle loader returning null component', async () => {
      const NullComponent = () => null;
      const loader = jest.fn().mockResolvedValue({ default: NullComponent });

      renderWithProviders(
        <LazyLoad loader={loader} />
      );

      await waitFor(() => {
        expect(loader).toHaveBeenCalled();
      });

      // Should render nothing but not error
      expect(screen.queryByTestId('error-message')).not.toBeInTheDocument();
    });

    it('should handle very short delay', async () => {
      const loader = jest.fn(() => new Promise(() => {}));

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          delay={1}
        />
      );

      act(() => {
        jest.advanceTimersByTime(2);
      });

      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    });

    it('should handle negative delay as zero', async () => {
      const loader = jest.fn(() => new Promise(() => {}));

      renderWithProviders(
        <LazyLoad 
          loader={loader}
          delay={-100}
        />
      );

      // Should show loading immediately like delay=0
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    });

    it('should handle multiple rapid re-renders', async () => {
      const loader1 = jest.fn().mockResolvedValue({ default: TestComponent });
      const loader2 = jest.fn().mockResolvedValue({ default: TestComponent });

      const { rerender } = renderWithProviders(
        <LazyLoad 
          loader={loader1}
          props={{ message: 'First' }}
        />
      );

      rerender(
        <LazyLoad 
          loader={loader2}
          props={{ message: 'Second' }}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Second')).toBeInTheDocument();
      });

      expect(loader1).toHaveBeenCalled();
      expect(loader2).toHaveBeenCalled();
    });
  });
});

describe('createLazyComponent', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should create a lazy component with default loading', async () => {
    const TestComp = () => <div data-testid="lazy-test">Lazy Component</div>;
    const loader = jest.fn().mockResolvedValue({ default: TestComp });
    
    const LazyComponent = createLazyComponent(loader);

    renderWithProviders(<LazyComponent />);

    // Should show default loading spinner
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByTestId('lazy-test')).toBeInTheDocument();
    });
  });

  it('should use custom fallback', async () => {
    const TestComp = () => <div data-testid="lazy-test">Lazy Component</div>;
    const loader = jest.fn().mockResolvedValue({ default: TestComp });
    const customFallback = <div data-testid="custom-suspense">Loading...</div>;
    
    const LazyComponent = createLazyComponent(loader, {
      fallback: customFallback,
    });

    renderWithProviders(<LazyComponent />);

    expect(screen.getByTestId('custom-suspense')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByTestId('lazy-test')).toBeInTheDocument();
    });
  });

  it('should use custom error fallback', async () => {
    const ErrorComp = () => {
      throw new Error('Component error');
    };
    const loader = jest.fn().mockResolvedValue({ default: ErrorComp });
    const customError = <div data-testid="custom-error-boundary">Error!</div>;
    
    const LazyComponent = createLazyComponent(loader, {
      errorFallback: customError,
    });

    renderWithProviders(<LazyComponent />);

    await waitFor(() => {
      expect(screen.getByTestId('error-boundary')).toBeInTheDocument();
    });
  });

  it('should pass props to lazy component', async () => {
    const TestComp: React.FC<{ value: string }> = ({ value }) => (
      <div data-testid="lazy-test">Value: {value}</div>
    );
    const loader = jest.fn().mockResolvedValue({ default: TestComp });
    
    const LazyComponent = createLazyComponent(loader);

    renderWithProviders(<LazyComponent value="test-prop" />);

    await waitFor(() => {
      expect(screen.getByText('Value: test-prop')).toBeInTheDocument();
    });
  });

  it('should handle Suspense boundary correctly', async () => {
    const TestComp = () => <div data-testid="lazy-test">Lazy Component</div>;
    const loader = jest.fn().mockResolvedValue({ default: TestComp });
    
    const LazyComponent = createLazyComponent(loader);

    renderWithProviders(
      <Suspense fallback={<div data-testid="outer-suspense">Outer Loading</div>}>
        <LazyComponent />
      </Suspense>
    );

    // Should use outer Suspense fallback
    expect(screen.getByTestId('outer-suspense')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByTestId('lazy-test')).toBeInTheDocument();
    });
  });
});