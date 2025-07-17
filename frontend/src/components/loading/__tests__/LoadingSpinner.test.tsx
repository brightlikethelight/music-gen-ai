/**
 * LoadingSpinner Component Tests
 * 
 * Comprehensive tests for the LoadingSpinner component including:
 * - Different sizes and colors
 * - Loading text display
 * - Full screen and overlay modes
 * - Animations
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '../../../test-utils';
import { LoadingSpinner } from '../LoadingSpinner';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, animate, transition, ...props }: any) => {
      // Add data attributes to verify animation props
      return (
        <div 
          {...props}
          data-animate={JSON.stringify(animate)}
          data-transition={JSON.stringify(transition)}
        >
          {children}
        </div>
      );
    },
    p: ({ children, ...props }: any) => <p {...props}>{children}</p>,
  },
}));

describe('LoadingSpinner', () => {
  describe('Basic Rendering', () => {
    it('should render loading spinner with default props', () => {
      renderWithProviders(<LoadingSpinner />);

      const spinner = screen.getByRole('status');
      expect(spinner).toBeInTheDocument();
      expect(spinner).toHaveAttribute('aria-label', 'Loading');
      expect(screen.getByText('Loading...')).toHaveClass('sr-only');
    });

    it('should render with loading text', () => {
      renderWithProviders(<LoadingSpinner text="Please wait..." />);

      expect(screen.getByText('Please wait...')).toBeInTheDocument();
      expect(screen.getByText('Please wait...')).not.toHaveClass('sr-only');
    });

    it('should apply custom className', () => {
      renderWithProviders(<LoadingSpinner className="custom-class mt-4" />);

      const container = screen.getByRole('status').parentElement;
      expect(container).toHaveClass('custom-class', 'mt-4');
    });
  });

  describe('Size Variants', () => {
    it('should render small size', () => {
      renderWithProviders(<LoadingSpinner size="sm" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('h-4', 'w-4');
    });

    it('should render medium size (default)', () => {
      renderWithProviders(<LoadingSpinner size="md" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('h-8', 'w-8');
    });

    it('should render large size', () => {
      renderWithProviders(<LoadingSpinner size="lg" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('h-12', 'w-12');
    });

    it('should render extra large size', () => {
      renderWithProviders(<LoadingSpinner size="xl" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('h-16', 'w-16');
    });
  });

  describe('Color Variants', () => {
    it('should render primary color (default)', () => {
      renderWithProviders(<LoadingSpinner color="primary" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('border-purple-600');
    });

    it('should render secondary color', () => {
      renderWithProviders(<LoadingSpinner color="secondary" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('border-blue-600');
    });

    it('should render white color', () => {
      renderWithProviders(<LoadingSpinner color="white" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('border-white');
    });

    it('should render gray color', () => {
      renderWithProviders(<LoadingSpinner color="gray" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('border-gray-400');
    });

    it('should apply correct text color based on spinner color', () => {
      renderWithProviders(<LoadingSpinner color="white" text="Loading..." />);

      expect(screen.getByText('Loading...')).toHaveClass('text-white');
    });
  });

  describe('Full Screen Mode', () => {
    it('should render in full screen mode', () => {
      renderWithProviders(<LoadingSpinner fullScreen />);

      const fullScreenContainer = screen.getByRole('status').parentElement?.parentElement;
      expect(fullScreenContainer).toHaveClass('fixed', 'inset-0', 'z-50');
    });

    it('should center spinner in full screen mode', () => {
      renderWithProviders(<LoadingSpinner fullScreen />);

      const fullScreenContainer = screen.getByRole('status').parentElement?.parentElement;
      expect(fullScreenContainer).toHaveClass('flex', 'items-center', 'justify-center');
    });

    it('should have dark background in full screen mode', () => {
      renderWithProviders(<LoadingSpinner fullScreen />);

      const fullScreenContainer = screen.getByRole('status').parentElement?.parentElement;
      expect(fullScreenContainer).toHaveClass('bg-gray-900');
    });

    it('should work with text in full screen mode', () => {
      renderWithProviders(<LoadingSpinner fullScreen text="Loading application..." />);

      expect(screen.getByText('Loading application...')).toBeInTheDocument();
    });
  });

  describe('Overlay Mode', () => {
    it('should render in overlay mode', () => {
      renderWithProviders(<LoadingSpinner overlay />);

      const overlayContainer = screen.getByRole('status').parentElement?.parentElement;
      expect(overlayContainer).toHaveClass('absolute', 'inset-0', 'z-40');
    });

    it('should have semi-transparent backdrop in overlay mode', () => {
      renderWithProviders(<LoadingSpinner overlay />);

      const overlayContainer = screen.getByRole('status').parentElement?.parentElement;
      expect(overlayContainer).toHaveClass('bg-gray-900/75', 'backdrop-blur-sm');
    });

    it('should have rounded corners in overlay mode', () => {
      renderWithProviders(<LoadingSpinner overlay />);

      const overlayContainer = screen.getByRole('status').parentElement?.parentElement;
      expect(overlayContainer).toHaveClass('rounded-lg');
    });
  });

  describe('Animation', () => {
    it('should have rotation animation', () => {
      renderWithProviders(<LoadingSpinner />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveAttribute('data-animate', JSON.stringify({ rotate: 360 }));
    });

    it('should have continuous linear animation', () => {
      renderWithProviders(<LoadingSpinner />);

      const spinner = screen.getByRole('status');
      const transition = JSON.parse(spinner.getAttribute('data-transition') || '{}');
      
      expect(transition).toEqual({
        duration: 1,
        repeat: Infinity,
        ease: 'linear',
      });
    });

    it('should animate text appearance', () => {
      renderWithProviders(<LoadingSpinner text="Loading..." />);

      const text = screen.getByText('Loading...');
      expect(text.tagName.toLowerCase()).toBe('p');
    });
  });

  describe('Combinations', () => {
    it('should handle size and color combinations', () => {
      renderWithProviders(<LoadingSpinner size="xl" color="secondary" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('h-16', 'w-16', 'border-blue-600');
    });

    it('should handle full screen with custom size', () => {
      renderWithProviders(<LoadingSpinner fullScreen size="xl" text="Loading..." />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('h-16', 'w-16');
      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    it('should handle overlay with custom color', () => {
      renderWithProviders(<LoadingSpinner overlay color="white" />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('border-white');
    });

    it('should not render both fullScreen and overlay', () => {
      renderWithProviders(<LoadingSpinner fullScreen overlay />);

      // fullScreen should take precedence
      const container = screen.getByRole('status').parentElement?.parentElement;
      expect(container).toHaveClass('fixed', 'inset-0', 'z-50');
      expect(container).not.toHaveClass('absolute', 'z-40');
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA attributes', () => {
      renderWithProviders(<LoadingSpinner />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveAttribute('aria-label', 'Loading');
    });

    it('should have screen reader only text', () => {
      renderWithProviders(<LoadingSpinner />);

      const srText = screen.getByText('Loading...');
      expect(srText).toHaveClass('sr-only');
    });

    it('should not hide visible loading text from screen readers', () => {
      renderWithProviders(<LoadingSpinner text="Please wait..." />);

      const visibleText = screen.getByText('Please wait...');
      expect(visibleText).not.toHaveClass('sr-only');
      
      // Screen reader text should still be present
      const srText = screen.getByText('Loading...');
      expect(srText).toHaveClass('sr-only');
    });

    it('should maintain semantic structure', () => {
      renderWithProviders(<LoadingSpinner text="Loading data..." />);

      const container = screen.getByRole('status').parentElement;
      expect(container).toHaveClass('flex', 'flex-col', 'items-center', 'justify-center');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty text string', () => {
      renderWithProviders(<LoadingSpinner text="" />);

      // Should not render empty paragraph
      const paragraphs = screen.queryAllByRole('paragraph');
      expect(paragraphs).toHaveLength(0);
    });

    it('should handle very long text', () => {
      const longText = 'A'.repeat(100);
      renderWithProviders(<LoadingSpinner text={longText} />);

      expect(screen.getByText(longText)).toBeInTheDocument();
    });

    it('should handle undefined props gracefully', () => {
      renderWithProviders(<LoadingSpinner size={undefined} color={undefined} />);

      const spinner = screen.getByRole('status');
      // Should use defaults
      expect(spinner).toHaveClass('h-8', 'w-8', 'border-purple-600');
    });

    it('should handle className when undefined', () => {
      renderWithProviders(<LoadingSpinner className={undefined} />);

      const container = screen.getByRole('status').parentElement;
      expect(container).toHaveClass('flex', 'flex-col', 'items-center', 'justify-center');
    });
  });

  describe('Styling', () => {
    it('should have transparent top border for rotation effect', () => {
      renderWithProviders(<LoadingSpinner />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('border-t-transparent');
    });

    it('should have rounded shape', () => {
      renderWithProviders(<LoadingSpinner />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('rounded-full');
    });

    it('should have consistent border width', () => {
      renderWithProviders(<LoadingSpinner />);

      const spinner = screen.getByRole('status');
      expect(spinner).toHaveClass('border-2');
    });

    it('should maintain aspect ratio', () => {
      const sizes = ['sm', 'md', 'lg', 'xl'] as const;
      
      sizes.forEach(size => {
        const { container } = renderWithProviders(<LoadingSpinner size={size} />);
        const spinner = container.querySelector('[role="status"]');
        
        // Width and height classes should match
        const classes = spinner?.className || '';
        const widthMatch = classes.match(/w-(\d+)/);
        const heightMatch = classes.match(/h-(\d+)/);
        
        expect(widthMatch?.[1]).toBe(heightMatch?.[1]);
      });
    });
  });

  describe('Text Styling', () => {
    it('should apply correct text size', () => {
      renderWithProviders(<LoadingSpinner text="Loading..." />);

      const text = screen.getByText('Loading...');
      expect(text).toHaveClass('text-sm');
    });

    it('should have proper spacing between spinner and text', () => {
      renderWithProviders(<LoadingSpinner text="Loading..." />);

      const text = screen.getByText('Loading...');
      expect(text).toHaveClass('mt-3');
    });

    it('should apply gray text color by default', () => {
      renderWithProviders(<LoadingSpinner text="Loading..." />);

      const text = screen.getByText('Loading...');
      expect(text).toHaveClass('text-gray-400');
    });
  });

  describe('Container Behavior', () => {
    it('should center content in default mode', () => {
      renderWithProviders(<LoadingSpinner />);

      const container = screen.getByRole('status').parentElement;
      expect(container).toHaveClass('flex', 'flex-col', 'items-center', 'justify-center');
    });

    it('should not interfere with parent layout in default mode', () => {
      renderWithProviders(
        <div data-testid="parent" className="flex">
          <LoadingSpinner />
        </div>
      );

      const parent = screen.getByTestId('parent');
      const spinner = screen.getByRole('status').parentElement;
      
      // Spinner container should not have position classes in default mode
      expect(spinner).not.toHaveClass('fixed', 'absolute');
    });
  });
});