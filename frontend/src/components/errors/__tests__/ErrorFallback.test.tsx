/**
 * ErrorFallback Component Tests
 * 
 * Comprehensive tests for the ErrorFallback component including:
 * - Default and minimal rendering modes
 * - Error display
 * - Technical details toggle
 * - Copy to clipboard functionality
 * - Reset and navigation actions
 * - Responsive design
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { ErrorFallback } from '../ErrorFallback';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock navigator.clipboard
const mockClipboard = {
  writeText: jest.fn(),
};
Object.defineProperty(navigator, 'clipboard', {
  value: mockClipboard,
  configurable: true,
});

// Mock window.location
const mockLocation = {
  href: 'http://localhost:3000/test-page',
};
Object.defineProperty(window, 'location', {
  value: mockLocation,
  writable: true,
});

describe('ErrorFallback', () => {
  const user = userEvent.setup();
  const mockReset = jest.fn();

  const defaultError = new Error('Something went wrong');
  defaultError.stack = `Error: Something went wrong
    at TestComponent (test.tsx:10:5)
    at ErrorBoundary (ErrorBoundary.tsx:20:10)`;

  const defaultErrorInfo = {
    componentStack: `
    at TestComponent
    at ErrorBoundary
    at App`,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockClipboard.writeText.mockResolvedValue(undefined);
  });

  describe('Default Rendering', () => {
    it('should render error fallback UI', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      expect(screen.getByText(/oops! something went wrong/i)).toBeInTheDocument();
      expect(screen.getByText(/we encountered an unexpected error/i)).toBeInTheDocument();
      expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
    });

    it('should render action buttons', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /go to home/i })).toBeInTheDocument();
    });

    it('should render support contact information', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      expect(screen.getByText(/if this problem persists/i)).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /support@musicgen.ai/i })).toBeInTheDocument();
    });
  });

  describe('Minimal Mode', () => {
    it('should render minimal UI when minimal prop is true', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          reset={mockReset} 
          minimal={true}
        />
      );

      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();

      // Should not render full UI elements
      expect(screen.queryByText(/oops! something went wrong/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/technical details/i)).not.toBeInTheDocument();
    });

    it('should still handle reset in minimal mode', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          reset={mockReset} 
          minimal={true}
        />
      );

      await user.click(screen.getByRole('button', { name: /try again/i }));
      expect(mockReset).toHaveBeenCalled();
    });
  });

  describe('Error Display', () => {
    it('should display error message', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('should handle errors without message', () => {
      const errorWithoutMessage = new Error();
      
      renderWithProviders(
        <ErrorFallback 
          error={errorWithoutMessage} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      expect(screen.getByText('Unknown error occurred')).toBeInTheDocument();
    });

    it('should handle null error info', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={null} 
          reset={mockReset} 
        />
      );

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });
  });

  describe('Technical Details', () => {
    it('should toggle technical details visibility', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      // Initially hidden
      expect(screen.queryByText(/stack trace/i)).not.toBeInTheDocument();

      // Click to show
      await user.click(screen.getByRole('button', { name: /technical details/i }));
      expect(screen.getByText(/stack trace/i)).toBeInTheDocument();
      expect(screen.getByText(/component stack/i)).toBeInTheDocument();

      // Click to hide
      await user.click(screen.getByRole('button', { name: /technical details/i }));
      await waitFor(() => {
        expect(screen.queryByText(/stack trace/i)).not.toBeInTheDocument();
      });
    });

    it('should display stack trace when expanded', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /technical details/i }));

      expect(screen.getByText(/Error: Something went wrong/)).toBeInTheDocument();
      expect(screen.getByText(/at TestComponent/)).toBeInTheDocument();
    });

    it('should display component stack when available', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /technical details/i }));

      const componentStackSection = screen.getByText(/component stack/i).parentElement;
      expect(componentStackSection).toHaveTextContent('at TestComponent');
      expect(componentStackSection).toHaveTextContent('at ErrorBoundary');
    });

    it('should handle missing stack trace', async () => {
      const errorWithoutStack = new Error('No stack');
      errorWithoutStack.stack = undefined;

      renderWithProviders(
        <ErrorFallback 
          error={errorWithoutStack} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /technical details/i }));
      expect(screen.getByText('No stack trace available')).toBeInTheDocument();
    });

    it('should display additional info', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /technical details/i }));

      expect(screen.getByText(/time:/i)).toBeInTheDocument();
      expect(screen.getByText(/url:/i)).toBeInTheDocument();
      expect(screen.getByText(/user agent:/i)).toBeInTheDocument();
    });
  });

  describe('Copy to Clipboard', () => {
    it('should copy error details to clipboard', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /technical details/i }));
      await user.click(screen.getByRole('button', { name: /copy error details/i }));

      expect(mockClipboard.writeText).toHaveBeenCalledWith(
        expect.stringContaining('Error: Something went wrong')
      );
      expect(mockClipboard.writeText).toHaveBeenCalledWith(
        expect.stringContaining('Stack Trace:')
      );
      expect(mockClipboard.writeText).toHaveBeenCalledWith(
        expect.stringContaining('Component Stack:')
      );
    });

    it('should show success feedback after copying', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /technical details/i }));
      await user.click(screen.getByRole('button', { name: /copy error details/i }));

      // Should show check icon
      const copyButton = screen.getByRole('button', { name: /copy error details/i });
      expect(copyButton.querySelector('.text-green-400')).toBeInTheDocument();

      // Should revert after 2 seconds
      await waitFor(() => {
        expect(copyButton.querySelector('.text-green-400')).not.toBeInTheDocument();
      }, { timeout: 3000 });
    });

    it('should handle clipboard API errors', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      mockClipboard.writeText.mockRejectedValueOnce(new Error('Clipboard error'));

      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /technical details/i }));
      await user.click(screen.getByRole('button', { name: /copy error details/i }));

      expect(consoleSpy).toHaveBeenCalledWith('Failed to copy error:', expect.any(Error));
      consoleSpy.mockRestore();
    });
  });

  describe('Actions', () => {
    it('should call reset function when Try Again is clicked', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /try again/i }));
      expect(mockReset).toHaveBeenCalledTimes(1);
    });

    it('should reset state before calling reset', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      // Expand details and copy
      await user.click(screen.getByRole('button', { name: /technical details/i }));
      expect(screen.getByText(/stack trace/i)).toBeInTheDocument();

      // Click reset
      await user.click(screen.getByRole('button', { name: /try again/i }));

      // State should be reset
      expect(mockReset).toHaveBeenCalled();
    });

    it('should navigate to home when Go to Home is clicked', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      await user.click(screen.getByRole('button', { name: /go to home/i }));
      expect(window.location.href).toBe('/');
    });
  });

  describe('Icons and Styling', () => {
    it('should render error icon', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      const icon = screen.getByRole('img', { hidden: true });
      expect(icon).toHaveClass('text-red-400');
    });

    it('should render appropriate icons for actions', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      const tryAgainButton = screen.getByRole('button', { name: /try again/i });
      expect(tryAgainButton.querySelector('svg')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA attributes', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      expect(screen.getByRole('heading', { name: /oops! something went wrong/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
    });

    it('should be keyboard navigable', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      // Tab through interactive elements
      await user.tab();
      expect(screen.getByRole('button', { name: /try again/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /go to home/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /technical details/i })).toHaveFocus();
    });

    it('should handle keyboard activation', async () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      const tryAgainButton = screen.getByRole('button', { name: /try again/i });
      tryAgainButton.focus();
      
      await user.keyboard('{Enter}');
      expect(mockReset).toHaveBeenCalled();
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long error messages', () => {
      const longError = new Error('A'.repeat(1000));
      
      renderWithProviders(
        <ErrorFallback 
          error={longError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      const errorMessage = screen.getByText('A'.repeat(1000));
      expect(errorMessage).toHaveClass('break-words');
    });

    it('should handle errors without errorInfo', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          reset={mockReset} 
        />
      );

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('should handle non-Error objects', () => {
      const stringError = { message: 'String error', stack: 'String stack' } as any;
      
      renderWithProviders(
        <ErrorFallback 
          error={stringError} 
          reset={mockReset} 
        />
      );

      expect(screen.getByText('String error')).toBeInTheDocument();
    });

    it('should handle server-side rendering', () => {
      // Mock window as undefined
      const originalWindow = global.window;
      // @ts-ignore
      delete global.window;

      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      expect(screen.getByText('Something went wrong')).toBeInTheDocument();

      global.window = originalWindow;
    });
  });

  describe('Responsive Design', () => {
    it('should handle mobile layout', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      const actionButtons = screen.getByRole('button', { name: /try again/i }).parentElement;
      expect(actionButtons).toHaveClass('flex-col', 'sm:flex-row');
    });

    it('should constrain width on large screens', () => {
      renderWithProviders(
        <ErrorFallback 
          error={defaultError} 
          errorInfo={defaultErrorInfo} 
          reset={mockReset} 
        />
      );

      const container = screen.getByText(/oops! something went wrong/i).closest('.max-w-2xl');
      expect(container).toBeInTheDocument();
    });
  });
});