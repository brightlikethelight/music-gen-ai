/**
 * ErrorMessage Component Tests
 * 
 * Comprehensive tests for the ErrorMessage component including:
 * - Different severity levels
 * - Title and message display
 * - Dismissible functionality
 * - Custom children content
 * - Icons and styling
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { ErrorMessage, ErrorSeverity } from '../ErrorMessage';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
}));

describe('ErrorMessage', () => {
  const user = userEvent.setup();
  const mockOnDismiss = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('should render error message with default severity', () => {
      renderWithProviders(
        <ErrorMessage message="This is an error message" />
      );

      expect(screen.getByText('This is an error message')).toBeInTheDocument();
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('should render with title', () => {
      renderWithProviders(
        <ErrorMessage 
          title="Error Title"
          message="This is an error message" 
        />
      );

      expect(screen.getByText('Error Title')).toBeInTheDocument();
      expect(screen.getByText('This is an error message')).toBeInTheDocument();
    });

    it('should render with custom children', () => {
      renderWithProviders(
        <ErrorMessage message="Main message">
          <button>Custom Action</button>
          <p>Additional information</p>
        </ErrorMessage>
      );

      expect(screen.getByText('Main message')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Custom Action' })).toBeInTheDocument();
      expect(screen.getByText('Additional information')).toBeInTheDocument();
    });

    it('should apply custom className', () => {
      renderWithProviders(
        <ErrorMessage 
          message="Test message" 
          className="custom-class mt-4"
        />
      );

      const alert = screen.getByRole('alert');
      expect(alert).toHaveClass('custom-class', 'mt-4');
    });
  });

  describe('Severity Levels', () => {
    const severityTests: Array<{ severity: ErrorSeverity; expectedClasses: string[] }> = [
      {
        severity: 'error',
        expectedClasses: ['bg-red-900/20', 'border-red-700', 'text-red-200'],
      },
      {
        severity: 'warning',
        expectedClasses: ['bg-yellow-900/20', 'border-yellow-700', 'text-yellow-200'],
      },
      {
        severity: 'info',
        expectedClasses: ['bg-blue-900/20', 'border-blue-700', 'text-blue-200'],
      },
      {
        severity: 'success',
        expectedClasses: ['bg-green-900/20', 'border-green-700', 'text-green-200'],
      },
    ];

    severityTests.forEach(({ severity, expectedClasses }) => {
      it(`should render with ${severity} severity styling`, () => {
        renderWithProviders(
          <ErrorMessage 
            message={`This is a ${severity} message`}
            severity={severity}
          />
        );

        const alert = screen.getByRole('alert');
        expectedClasses.forEach(className => {
          expect(alert).toHaveClass(className);
        });
      });

      it(`should render correct icon for ${severity} severity`, () => {
        renderWithProviders(
          <ErrorMessage 
            message={`This is a ${severity} message`}
            severity={severity}
          />
        );

        const icon = screen.getByRole('alert').querySelector('svg');
        expect(icon).toBeInTheDocument();
        
        // Check icon color class
        const iconColorClass = severity === 'error' ? 'text-red-400' :
                              severity === 'warning' ? 'text-yellow-400' :
                              severity === 'info' ? 'text-blue-400' :
                              'text-green-400';
        expect(icon).toHaveClass(iconColorClass);
      });
    });

    it('should apply correct title color based on severity', () => {
      renderWithProviders(
        <ErrorMessage 
          title="Warning Title"
          message="Warning message"
          severity="warning"
        />
      );

      const title = screen.getByText('Warning Title');
      expect(title).toHaveClass('text-yellow-300');
    });
  });

  describe('Dismissible Functionality', () => {
    it('should show dismiss button when dismissible and onDismiss provided', () => {
      renderWithProviders(
        <ErrorMessage 
          message="Dismissible message"
          onDismiss={mockOnDismiss}
          dismissible={true}
        />
      );

      expect(screen.getByRole('button', { name: 'Dismiss' })).toBeInTheDocument();
    });

    it('should not show dismiss button when dismissible is false', () => {
      renderWithProviders(
        <ErrorMessage 
          message="Non-dismissible message"
          onDismiss={mockOnDismiss}
          dismissible={false}
        />
      );

      expect(screen.queryByRole('button', { name: 'Dismiss' })).not.toBeInTheDocument();
    });

    it('should not show dismiss button when onDismiss is not provided', () => {
      renderWithProviders(
        <ErrorMessage 
          message="Message without handler"
          dismissible={true}
        />
      );

      expect(screen.queryByRole('button', { name: 'Dismiss' })).not.toBeInTheDocument();
    });

    it('should call onDismiss when dismiss button is clicked', async () => {
      renderWithProviders(
        <ErrorMessage 
          message="Dismissible message"
          onDismiss={mockOnDismiss}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Dismiss' }));
      expect(mockOnDismiss).toHaveBeenCalledTimes(1);
    });

    it('should handle keyboard dismiss', async () => {
      renderWithProviders(
        <ErrorMessage 
          message="Dismissible message"
          onDismiss={mockOnDismiss}
        />
      );

      const dismissButton = screen.getByRole('button', { name: 'Dismiss' });
      dismissButton.focus();
      
      await user.keyboard('{Enter}');
      expect(mockOnDismiss).toHaveBeenCalledTimes(1);
    });
  });

  describe('Icons', () => {
    it('should render error icon for error severity', () => {
      renderWithProviders(
        <ErrorMessage message="Error" severity="error" />
      );

      const icon = screen.getByRole('alert').querySelector('svg');
      expect(icon).toHaveClass('text-red-400');
    });

    it('should render warning icon for warning severity', () => {
      renderWithProviders(
        <ErrorMessage message="Warning" severity="warning" />
      );

      const icon = screen.getByRole('alert').querySelector('svg');
      expect(icon).toHaveClass('text-yellow-400');
    });

    it('should render info icon for info severity', () => {
      renderWithProviders(
        <ErrorMessage message="Info" severity="info" />
      );

      const icon = screen.getByRole('alert').querySelector('svg');
      expect(icon).toHaveClass('text-blue-400');
    });

    it('should render success icon for success severity', () => {
      renderWithProviders(
        <ErrorMessage message="Success" severity="success" />
      );

      const icon = screen.getByRole('alert').querySelector('svg');
      expect(icon).toHaveClass('text-green-400');
    });
  });

  describe('Animations', () => {
    it('should have initial animation properties', () => {
      renderWithProviders(
        <ErrorMessage message="Animated message" />
      );

      const alert = screen.getByRole('alert');
      // Motion div should have been rendered (mocked as regular div)
      expect(alert).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have role="alert" for screen readers', () => {
      renderWithProviders(
        <ErrorMessage message="Alert message" />
      );

      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('should have aria-live="polite"', () => {
      renderWithProviders(
        <ErrorMessage message="Live message" />
      );

      expect(screen.getByRole('alert')).toHaveAttribute('aria-live', 'polite');
    });

    it('should have accessible dismiss button', () => {
      renderWithProviders(
        <ErrorMessage 
          message="Dismissible"
          onDismiss={mockOnDismiss}
        />
      );

      const dismissButton = screen.getByRole('button', { name: 'Dismiss' });
      expect(dismissButton).toHaveAttribute('aria-label', 'Dismiss');
    });

    it('should hide decorative icons from screen readers', () => {
      renderWithProviders(
        <ErrorMessage message="Message with icon" />
      );

      const icon = screen.getByRole('alert').querySelector('svg');
      expect(icon).toHaveAttribute('aria-hidden', 'true');
    });

    it('should be keyboard navigable', async () => {
      renderWithProviders(
        <ErrorMessage 
          message="Keyboard test"
          onDismiss={mockOnDismiss}
        />
      );

      await user.tab();
      expect(screen.getByRole('button', { name: 'Dismiss' })).toHaveFocus();
    });
  });

  describe('Complex Content', () => {
    it('should render multiple children elements', () => {
      renderWithProviders(
        <ErrorMessage message="Main error">
          <ul>
            <li>Error detail 1</li>
            <li>Error detail 2</li>
            <li>Error detail 3</li>
          </ul>
          <button>Retry</button>
        </ErrorMessage>
      );

      expect(screen.getByText('Main error')).toBeInTheDocument();
      expect(screen.getByText('Error detail 1')).toBeInTheDocument();
      expect(screen.getByText('Error detail 2')).toBeInTheDocument();
      expect(screen.getByText('Error detail 3')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    });

    it('should handle React components as children', () => {
      const CustomComponent = () => <div data-testid="custom">Custom content</div>;

      renderWithProviders(
        <ErrorMessage message="Parent message">
          <CustomComponent />
        </ErrorMessage>
      );

      expect(screen.getByTestId('custom')).toBeInTheDocument();
      expect(screen.getByText('Custom content')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty message', () => {
      renderWithProviders(
        <ErrorMessage message="" />
      );

      const alert = screen.getByRole('alert');
      expect(alert).toBeInTheDocument();
    });

    it('should handle very long messages', () => {
      const longMessage = 'A'.repeat(500);
      
      renderWithProviders(
        <ErrorMessage message={longMessage} />
      );

      expect(screen.getByText(longMessage)).toBeInTheDocument();
    });

    it('should handle special characters in message', () => {
      const specialMessage = 'Error: <script>alert("XSS")</script> & special chars';
      
      renderWithProviders(
        <ErrorMessage message={specialMessage} />
      );

      expect(screen.getByText(specialMessage)).toBeInTheDocument();
    });

    it('should handle rapid dismiss clicks', async () => {
      renderWithProviders(
        <ErrorMessage 
          message="Rapid clicks"
          onDismiss={mockOnDismiss}
        />
      );

      const dismissButton = screen.getByRole('button', { name: 'Dismiss' });
      
      await user.click(dismissButton);
      await user.click(dismissButton);
      await user.click(dismissButton);

      // Should only call once per click
      expect(mockOnDismiss).toHaveBeenCalledTimes(3);
    });

    it('should handle undefined severity gracefully', () => {
      renderWithProviders(
        <ErrorMessage 
          message="Default severity"
          severity={undefined}
        />
      );

      const alert = screen.getByRole('alert');
      expect(alert).toHaveClass('bg-red-900/20'); // Default to error styling
    });
  });

  describe('Styling Combinations', () => {
    it('should combine severity styling with custom className', () => {
      renderWithProviders(
        <ErrorMessage 
          message="Styled message"
          severity="warning"
          className="mt-8 shadow-lg"
        />
      );

      const alert = screen.getByRole('alert');
      expect(alert).toHaveClass('bg-yellow-900/20', 'border-yellow-700', 'mt-8', 'shadow-lg');
    });

    it('should maintain consistent padding and spacing', () => {
      renderWithProviders(
        <ErrorMessage 
          title="Title"
          message="Message"
          onDismiss={mockOnDismiss}
        >
          <div>Child content</div>
        </ErrorMessage>
      );

      const alert = screen.getByRole('alert');
      expect(alert).toHaveClass('p-4');
      
      const contentContainer = screen.getByText('Message').parentElement;
      expect(contentContainer).toHaveClass('ml-3', 'flex-1');
    });
  });

  describe('Focus Management', () => {
    it('should have focusable dismiss button', () => {
      renderWithProviders(
        <ErrorMessage 
          message="Focus test"
          onDismiss={mockOnDismiss}
        />
      );

      const dismissButton = screen.getByRole('button', { name: 'Dismiss' });
      expect(dismissButton).toHaveClass('focus:outline-none', 'focus:ring-2');
    });

    it('should show focus ring on dismiss button', async () => {
      renderWithProviders(
        <ErrorMessage 
          message="Focus ring test"
          onDismiss={mockOnDismiss}
        />
      );

      const dismissButton = screen.getByRole('button', { name: 'Dismiss' });
      dismissButton.focus();
      
      expect(dismissButton).toHaveClass('focus:ring-2', 'focus:ring-offset-2');
    });
  });

  describe('Responsive Design', () => {
    it('should maintain layout on small screens', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      renderWithProviders(
        <ErrorMessage 
          title="Mobile Title"
          message="Mobile message"
          onDismiss={mockOnDismiss}
        />
      );

      const alert = screen.getByRole('alert');
      expect(alert).toHaveClass('flex');
    });

    it('should handle long content without breaking layout', () => {
      renderWithProviders(
        <ErrorMessage 
          title="A".repeat(100)
          message="B".repeat(200)
          onDismiss={mockOnDismiss}
        />
      );

      const alert = screen.getByRole('alert');
      expect(alert).toBeInTheDocument();
      
      // Content should wrap properly
      const messageContainer = screen.getByText('B'.repeat(200)).parentElement;
      expect(messageContainer).toHaveClass('flex-1');
    });
  });
});