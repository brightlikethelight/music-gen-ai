/**
 * SkeletonLoader Component Tests
 * 
 * Comprehensive tests for the SkeletonLoader component including:
 * - Different skeleton variants
 * - Custom dimensions
 * - Multiple skeletons
 * - Animation control
 * - Shimmer effect
 * - Responsive behavior
 * - Edge cases
 */

import React from 'react';
import { screen, within } from '@testing-library/react';
import { renderWithProviders } from '../../../test-utils';
import { SkeletonLoader } from '../SkeletonLoader';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, variants, initial, animate, style, ...props }: any) => (
      <div 
        {...props}
        style={style}
        data-variants={variants ? 'shimmer' : undefined}
        data-initial={initial}
        data-animate={animate}
      >
        {children}
      </div>
    ),
  },
}));

describe('SkeletonLoader', () => {
  describe('Text Variant', () => {
    it('should render text skeleton with default width', () => {
      renderWithProviders(<SkeletonLoader variant="text" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveClass('h-4', 'rounded');
      expect(skeleton).toHaveStyle({ width: '100%' });
    });

    it('should render text skeleton with custom width', () => {
      renderWithProviders(<SkeletonLoader variant="text" width="200px" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveStyle({ width: '200px' });
    });

    it('should render text skeleton with percentage width', () => {
      renderWithProviders(<SkeletonLoader variant="text" width="75%" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveStyle({ width: '75%' });
    });

    it('should apply custom className', () => {
      renderWithProviders(<SkeletonLoader variant="text" className="custom-class" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveClass('custom-class');
    });
  });

  describe('Circular Variant', () => {
    it('should render circular skeleton with default size', () => {
      renderWithProviders(<SkeletonLoader variant="circular" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveClass('rounded-full');
      expect(skeleton).toHaveStyle({ width: '40px', height: '40px' });
    });

    it('should render circular skeleton with custom size', () => {
      renderWithProviders(<SkeletonLoader variant="circular" width={60} height={60} />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveStyle({ width: '60px', height: '60px' });
    });

    it('should use width for height if height not specified', () => {
      renderWithProviders(<SkeletonLoader variant="circular" width={80} />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveStyle({ width: '80px', height: '80px' });
    });
  });

  describe('Rectangular Variant', () => {
    it('should render rectangular skeleton with default dimensions', () => {
      renderWithProviders(<SkeletonLoader variant="rectangular" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveClass('rounded');
      expect(skeleton).toHaveStyle({ width: '100%', height: '60px' });
    });

    it('should render rectangular skeleton with custom dimensions', () => {
      renderWithProviders(
        <SkeletonLoader variant="rectangular" width={300} height={100} />
      );

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveStyle({ width: '300px', height: '100px' });
    });
  });

  describe('Card Variant', () => {
    it('should render card skeleton with image and text placeholders', () => {
      renderWithProviders(<SkeletonLoader variant="card" />);

      const card = screen.getByRole('generic');
      expect(card).toHaveClass('bg-gray-800', 'rounded-lg', 'p-4');

      // Should have image placeholder
      const imagePlaceholder = card.querySelector('.h-32');
      expect(imagePlaceholder).toBeInTheDocument();
      expect(imagePlaceholder).toHaveClass('rounded', 'mb-4');

      // Should have two text placeholders
      const textPlaceholders = card.querySelectorAll('.h-4');
      expect(textPlaceholders).toHaveLength(2);
      expect(textPlaceholders[0]).toHaveStyle({ width: '70%' });
      expect(textPlaceholders[1]).toHaveStyle({ width: '50%' });
    });

    it('should apply custom className to card', () => {
      renderWithProviders(<SkeletonLoader variant="card" className="shadow-lg" />);

      const card = screen.getByRole('generic');
      expect(card).toHaveClass('shadow-lg');
    });
  });

  describe('List Variant', () => {
    it('should render list skeleton with default count', () => {
      renderWithProviders(<SkeletonLoader variant="list" />);

      const list = screen.getByRole('generic');
      const items = list.querySelectorAll('.flex.items-center');
      expect(items).toHaveLength(1);
    });

    it('should render multiple list items based on count', () => {
      renderWithProviders(<SkeletonLoader variant="list" count={5} />);

      const list = screen.getByRole('generic');
      const items = list.querySelectorAll('.flex.items-center');
      expect(items).toHaveLength(5);
    });

    it('should render list items with avatar and text', () => {
      renderWithProviders(<SkeletonLoader variant="list" count={1} />);

      const item = screen.getByRole('generic').querySelector('.flex.items-center');
      
      // Avatar
      const avatar = item?.querySelector('.h-10.w-10.rounded-full');
      expect(avatar).toBeInTheDocument();

      // Text lines
      const textLines = item?.querySelectorAll('.rounded');
      expect(textLines?.length).toBeGreaterThanOrEqual(2);
    });

    it('should apply proper spacing between list items', () => {
      renderWithProviders(<SkeletonLoader variant="list" count={3} />);

      const list = screen.getByRole('generic');
      expect(list).toHaveClass('space-y-3');
    });
  });

  describe('Multiple Skeletons', () => {
    it('should render multiple text skeletons', () => {
      renderWithProviders(<SkeletonLoader variant="text" count={3} />);

      const container = screen.getByRole('generic');
      const skeletons = container.querySelectorAll('.h-4');
      expect(skeletons).toHaveLength(3);
    });

    it('should render multiple rectangular skeletons', () => {
      renderWithProviders(<SkeletonLoader variant="rectangular" count={4} />);

      const container = screen.getByRole('generic');
      const skeletons = container.querySelectorAll('.rounded');
      expect(skeletons).toHaveLength(4);
    });

    it('should add spacing between multiple skeletons', () => {
      renderWithProviders(<SkeletonLoader variant="text" count={2} />);

      const container = screen.getByRole('generic');
      expect(container).toHaveClass('space-y-2');
    });

    it('should not wrap list variant in additional container', () => {
      renderWithProviders(<SkeletonLoader variant="list" count={3} />);

      // List variant handles its own multiple items
      const lists = screen.getAllByRole('generic');
      expect(lists).toHaveLength(1);
    });
  });

  describe('Animation', () => {
    it('should apply shimmer animation by default', () => {
      renderWithProviders(<SkeletonLoader variant="text" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveAttribute('data-variants', 'shimmer');
      expect(skeleton).toHaveAttribute('data-initial', 'initial');
      expect(skeleton).toHaveAttribute('data-animate', 'animate');
    });

    it('should disable animation when animate is false', () => {
      renderWithProviders(<SkeletonLoader variant="text" animate={false} />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).not.toHaveAttribute('data-variants');
      expect(skeleton).toHaveAttribute('data-initial', 'undefined');
      expect(skeleton).toHaveAttribute('data-animate', 'undefined');
    });

    it('should apply gradient background for shimmer effect', () => {
      renderWithProviders(<SkeletonLoader variant="text" animate={true} />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveClass('bg-gradient-to-r', 'from-gray-700', 'via-gray-600', 'to-gray-700');
    });

    it('should not apply gradient when animation is disabled', () => {
      renderWithProviders(<SkeletonLoader variant="text" animate={false} />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).not.toHaveClass('bg-gradient-to-r');
    });
  });

  describe('Styling', () => {
    it('should apply base gray background', () => {
      renderWithProviders(<SkeletonLoader variant="text" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveClass('bg-gray-700');
    });

    it('should apply appropriate border radius for each variant', () => {
      const { rerender } = renderWithProviders(<SkeletonLoader variant="text" />);
      expect(screen.getByRole('generic')).toHaveClass('rounded');

      rerender(<SkeletonLoader variant="circular" />);
      expect(screen.getByRole('generic')).toHaveClass('rounded-full');

      rerender(<SkeletonLoader variant="rectangular" />);
      expect(screen.getByRole('generic')).toHaveClass('rounded');
    });

    it('should combine custom className with default classes', () => {
      renderWithProviders(
        <SkeletonLoader variant="text" className="mt-4 opacity-50" />
      );

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveClass('h-4', 'rounded', 'bg-gray-700', 'mt-4', 'opacity-50');
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero count', () => {
      renderWithProviders(<SkeletonLoader variant="text" count={0} />);

      // Should still render at least one
      const skeletons = screen.getAllByRole('generic');
      expect(skeletons).toHaveLength(1);
    });

    it('should handle negative count', () => {
      renderWithProviders(<SkeletonLoader variant="text" count={-1} />);

      // Should default to 1
      const skeletons = screen.getAllByRole('generic');
      expect(skeletons).toHaveLength(1);
    });

    it('should handle very large count gracefully', () => {
      renderWithProviders(<SkeletonLoader variant="text" count={100} />);

      const container = screen.getByRole('generic');
      const skeletons = container.querySelectorAll('.h-4');
      expect(skeletons).toHaveLength(100);
    });

    it('should handle string dimensions', () => {
      renderWithProviders(
        <SkeletonLoader variant="rectangular" width="300px" height="100px" />
      );

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveStyle({ width: '300px', height: '100px' });
    });

    it('should handle invalid variant gracefully', () => {
      // @ts-ignore - Testing invalid variant
      renderWithProviders(<SkeletonLoader variant="invalid" />);

      // Should not render anything for invalid variant
      expect(screen.queryByRole('generic')).not.toBeInTheDocument();
    });
  });

  describe('Responsive Behavior', () => {
    it('should use percentage width for responsive text skeleton', () => {
      renderWithProviders(<SkeletonLoader variant="text" width="100%" />);

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveStyle({ width: '100%' });
    });

    it('should maintain aspect ratio for circular skeleton', () => {
      renderWithProviders(<SkeletonLoader variant="circular" width={50} />);

      const skeleton = screen.getByRole('generic');
      const style = skeleton.getAttribute('style');
      expect(style).toContain('width: 50px');
      expect(style).toContain('height: 50px');
    });

    it('should allow responsive rectangular skeleton', () => {
      renderWithProviders(
        <SkeletonLoader variant="rectangular" width="100%" height={200} />
      );

      const skeleton = screen.getByRole('generic');
      expect(skeleton).toHaveStyle({ width: '100%', height: '200px' });
    });
  });

  describe('Complex Layouts', () => {
    it('should render card skeleton with proper structure', () => {
      renderWithProviders(<SkeletonLoader variant="card" />);

      const card = screen.getByRole('generic');
      
      // Check card structure
      expect(card.children).toHaveLength(3); // Image + 2 text lines
      
      // Check image placeholder
      expect(card.children[0]).toHaveClass('h-32', 'rounded', 'mb-4');
      
      // Check text placeholders
      expect(card.children[1]).toHaveClass('h-4', 'rounded', 'mb-2');
      expect(card.children[2]).toHaveClass('h-4', 'rounded');
    });

    it('should render list skeleton with proper item structure', () => {
      renderWithProviders(<SkeletonLoader variant="list" count={1} />);

      const list = screen.getByRole('generic');
      const item = list.querySelector('.flex.items-center');
      
      expect(item).toBeInTheDocument();
      
      // Check avatar
      const avatar = item?.querySelector('.h-10.w-10.rounded-full');
      expect(avatar).toBeInTheDocument();
      
      // Check text container
      const textContainer = item?.querySelector('.flex-1');
      expect(textContainer).toBeInTheDocument();
      expect(textContainer?.querySelectorAll('.rounded')).toHaveLength(2);
    });
  });

  describe('Performance', () => {
    it('should not re-render when count does not change', () => {
      const { rerender } = renderWithProviders(
        <SkeletonLoader variant="text" count={3} />
      );

      const container = screen.getByRole('generic');
      const initialSkeletons = container.querySelectorAll('.h-4');

      rerender(<SkeletonLoader variant="text" count={3} />);

      const afterSkeletons = container.querySelectorAll('.h-4');
      expect(afterSkeletons).toHaveLength(initialSkeletons.length);
    });

    it('should handle animation state changes', () => {
      const { rerender } = renderWithProviders(
        <SkeletonLoader variant="text" animate={true} />
      );

      expect(screen.getByRole('generic')).toHaveAttribute('data-variants', 'shimmer');

      rerender(<SkeletonLoader variant="text" animate={false} />);

      expect(screen.getByRole('generic')).not.toHaveAttribute('data-variants');
    });
  });
});