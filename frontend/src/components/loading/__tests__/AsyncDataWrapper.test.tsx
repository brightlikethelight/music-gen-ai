/**
 * AsyncDataWrapper Component Tests
 * 
 * Comprehensive tests for the AsyncDataWrapper component including:
 * - Loading states
 * - Error states
 * - Empty states
 * - Success states with data
 * - Custom components
 * - Skeleton loading
 * - Retry functionality
 * - Animations
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { AsyncDataWrapper } from '../AsyncDataWrapper';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock child components
jest.mock('../LoadingSpinner', () => ({
  LoadingSpinner: ({ size, text }: any) => (
    <div data-testid="loading-spinner" data-size={size} data-text={text}>
      {text || 'Loading...'}
    </div>
  ),
}));

jest.mock('../SkeletonLoader', () => ({
  SkeletonLoader: ({ variant, count, animate }: any) => (
    <div 
      data-testid="skeleton-loader" 
      data-variant={variant} 
      data-count={count}
      data-animate={animate}
    >
      Skeleton
    </div>
  ),
}));

jest.mock('../../errors/ErrorMessage', () => ({
  ErrorMessage: ({ severity, title, message, children }: any) => (
    <div data-testid="error-message" data-severity={severity}>
      <h3>{title}</h3>
      <p>{message}</p>
      {children}
    </div>
  ),
}));

// Sample data types for testing
interface TestData {
  id: number;
  name: string;
  value: string;
}

describe('AsyncDataWrapper', () => {
  const user = userEvent.setup();
  const mockRetryFn = jest.fn();

  const sampleData: TestData = {
    id: 1,
    name: 'Test Item',
    value: 'test-value',
  };

  const renderChild = (data: TestData) => (
    <div data-testid="data-content">
      <h2>{data.name}</h2>
      <p>{data.value}</p>
    </div>
  );

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Loading State', () => {
    it('should show default loading spinner when loading', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    it('should show custom loading component when provided', () => {
      const customLoader = <div data-testid="custom-loader">Custom Loading...</div>;

      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
          loadingComponent={customLoader}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('custom-loader')).toBeInTheDocument();
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });

    it('should show skeleton loader when showSkeleton is true', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
          showSkeleton={true}
          skeletonVariant="list"
          skeletonCount={5}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      const skeleton = screen.getByTestId('skeleton-loader');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveAttribute('data-variant', 'list');
      expect(skeleton).toHaveAttribute('data-count', '5');
      expect(skeleton).toHaveAttribute('data-animate', 'true');
    });

    it('should apply className to loading state', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
          className="custom-loading-class"
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      const container = screen.getByTestId('loading-spinner').parentElement;
      expect(container).toHaveClass('custom-loading-class');
    });
  });

  describe('Error State', () => {
    it('should show default error message when error occurs', () => {
      const error = new Error('Failed to load data');

      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={error}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      const errorMessage = screen.getByTestId('error-message');
      expect(errorMessage).toBeInTheDocument();
      expect(screen.getByText('Error Loading Data')).toBeInTheDocument();
      expect(screen.getByText('Failed to load data')).toBeInTheDocument();
    });

    it('should show string error message', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error="Network error occurred"
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('Network error occurred')).toBeInTheDocument();
    });

    it('should show custom error component when provided', () => {
      const customError = <div data-testid="custom-error">Custom Error UI</div>;

      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={new Error('Test error')}
          errorComponent={customError}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('custom-error')).toBeInTheDocument();
      expect(screen.queryByTestId('error-message')).not.toBeInTheDocument();
    });

    it('should show retry button when retryFn is provided', async () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={new Error('Test error')}
          retryFn={mockRetryFn}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      const retryButton = screen.getByRole('button', { name: /try again/i });
      expect(retryButton).toBeInTheDocument();

      await user.click(retryButton);
      expect(mockRetryFn).toHaveBeenCalledTimes(1);
    });

    it('should not show retry button when retryFn is not provided', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={new Error('Test error')}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.queryByRole('button', { name: /try again/i })).not.toBeInTheDocument();
    });
  });

  describe('Empty State', () => {
    it('should show default empty message for null data', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('No data available')).toBeInTheDocument();
    });

    it('should show default empty message for undefined data', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={undefined}
          isLoading={false}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('No data available')).toBeInTheDocument();
    });

    it('should show default empty message for empty array', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={[]}
          isLoading={false}
          error={null}
        >
          {(data: any[]) => <div>Items: {data.length}</div>}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('No data available')).toBeInTheDocument();
    });

    it('should show custom empty component when provided', () => {
      const customEmpty = <div data-testid="custom-empty">No items found</div>;

      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={null}
          emptyComponent={customEmpty}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('custom-empty')).toBeInTheDocument();
      expect(screen.queryByText('No data available')).not.toBeInTheDocument();
    });
  });

  describe('Success State', () => {
    it('should render children with data when loaded', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={sampleData}
          isLoading={false}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('data-content')).toBeInTheDocument();
      expect(screen.getByText('Test Item')).toBeInTheDocument();
      expect(screen.getByText('test-value')).toBeInTheDocument();
    });

    it('should render array data', () => {
      const arrayData = [
        { id: 1, name: 'Item 1' },
        { id: 2, name: 'Item 2' },
        { id: 3, name: 'Item 3' },
      ];

      renderWithProviders(
        <AsyncDataWrapper
          data={arrayData}
          isLoading={false}
          error={null}
        >
          {(data) => (
            <ul data-testid="data-list">
              {data.map((item) => (
                <li key={item.id}>{item.name}</li>
              ))}
            </ul>
          )}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('data-list')).toBeInTheDocument();
      expect(screen.getByText('Item 1')).toBeInTheDocument();
      expect(screen.getByText('Item 2')).toBeInTheDocument();
      expect(screen.getByText('Item 3')).toBeInTheDocument();
    });

    it('should handle non-empty arrays as success state', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={[{ id: 1 }]}
          isLoading={false}
          error={null}
        >
          {(data) => <div>Items count: {data.length}</div>}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('Items count: 1')).toBeInTheDocument();
      expect(screen.queryByText('No data available')).not.toBeInTheDocument();
    });
  });

  describe('State Transitions', () => {
    it('should transition from loading to success', async () => {
      const { rerender } = renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();

      rerender(
        <AsyncDataWrapper
          data={sampleData}
          isLoading={false}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      await waitFor(() => {
        expect(screen.getByTestId('data-content')).toBeInTheDocument();
        expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
      });
    });

    it('should transition from loading to error', async () => {
      const { rerender } = renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();

      rerender(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={new Error('Load failed')}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toBeInTheDocument();
        expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
      });
    });

    it('should transition from error to loading on retry', async () => {
      const { rerender } = renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={new Error('Test error')}
          retryFn={mockRetryFn}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      const retryButton = screen.getByRole('button', { name: /try again/i });
      await user.click(retryButton);

      rerender(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
          retryFn={mockRetryFn}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
      expect(screen.queryByTestId('error-message')).not.toBeInTheDocument();
    });
  });

  describe('Animations', () => {
    it('should apply initial animation to error state', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={new Error('Test error')}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      const errorContainer = screen.getByTestId('error-message').parentElement;
      expect(errorContainer).toHaveAttribute('initial', JSON.stringify({ opacity: 0 }));
      expect(errorContainer).toHaveAttribute('animate', JSON.stringify({ opacity: 1 }));
    });

    it('should apply initial animation to empty state', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      const emptyContainer = screen.getByText('No data available').parentElement;
      expect(emptyContainer).toHaveAttribute('initial', JSON.stringify({ opacity: 0 }));
      expect(emptyContainer).toHaveAttribute('animate', JSON.stringify({ opacity: 1 }));
    });

    it('should apply animation to success state', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={sampleData}
          isLoading={false}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      const contentContainer = screen.getByTestId('data-content').parentElement;
      expect(contentContainer).toHaveAttribute('initial', JSON.stringify({ opacity: 0, y: 10 }));
      expect(contentContainer).toHaveAttribute('animate', JSON.stringify({ opacity: 1, y: 0 }));
    });
  });

  describe('className Application', () => {
    it('should apply className to all states', () => {
      const className = 'custom-wrapper-class';

      // Loading state
      const { rerender } = renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
          className={className}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      let container = screen.getByTestId('loading-spinner').parentElement;
      expect(container).toHaveClass(className);

      // Error state
      rerender(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={new Error('Test')}
          className={className}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      container = screen.getByTestId('error-message').parentElement;
      expect(container).toHaveClass(className);

      // Empty state
      rerender(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={null}
          className={className}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      container = screen.getByText('No data available').parentElement;
      expect(container).toHaveClass(className);

      // Success state
      rerender(
        <AsyncDataWrapper
          data={sampleData}
          isLoading={false}
          error={null}
          className={className}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      container = screen.getByTestId('data-content').parentElement;
      expect(container).toHaveClass(className);
    });
  });

  describe('Edge Cases', () => {
    it('should handle complex nested data', () => {
      const complexData = {
        user: {
          id: 1,
          profile: {
            name: 'Test User',
            settings: {
              theme: 'dark',
            },
          },
        },
      };

      renderWithProviders(
        <AsyncDataWrapper
          data={complexData}
          isLoading={false}
          error={null}
        >
          {(data) => (
            <div data-testid="complex-data">
              {data.user.profile.name} - {data.user.profile.settings.theme}
            </div>
          )}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('Test User - dark')).toBeInTheDocument();
    });

    it('should handle falsy but valid data', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={0}
          isLoading={false}
          error={null}
        >
          {(data) => <div>Value: {data}</div>}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('Value: 0')).toBeInTheDocument();
    });

    it('should handle boolean false as valid data', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={false}
          isLoading={false}
          error={null}
        >
          {(data) => <div>Value: {data.toString()}</div>}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('Value: false')).toBeInTheDocument();
    });

    it('should handle empty string as empty state', () => {
      renderWithProviders(
        <AsyncDataWrapper
          data={''}
          isLoading={false}
          error={null}
        >
          {(data) => <div>Value: {data}</div>}
        </AsyncDataWrapper>
      );

      // Empty string is considered empty
      expect(screen.getByText('No data available')).toBeInTheDocument();
    });

    it('should handle very long error messages', () => {
      const longError = 'A'.repeat(500);

      renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={longError}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      expect(screen.getByText(longError)).toBeInTheDocument();
    });

    it('should handle rapid state changes', async () => {
      const { rerender } = renderWithProviders(
        <AsyncDataWrapper
          data={null}
          isLoading={true}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      // Rapidly change states
      rerender(
        <AsyncDataWrapper
          data={null}
          isLoading={false}
          error={new Error('Error')}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      rerender(
        <AsyncDataWrapper
          data={sampleData}
          isLoading={false}
          error={null}
        >
          {renderChild}
        </AsyncDataWrapper>
      );

      // Should show final state
      expect(screen.getByTestId('data-content')).toBeInTheDocument();
    });
  });

  describe('TypeScript Type Safety', () => {
    it('should maintain type safety for data in children', () => {
      interface TypedData {
        id: number;
        name: string;
        optional?: string;
      }

      const typedData: TypedData = {
        id: 1,
        name: 'Typed Item',
      };

      renderWithProviders(
        <AsyncDataWrapper<TypedData>
          data={typedData}
          isLoading={false}
          error={null}
        >
          {(data) => (
            <div data-testid="typed-content">
              ID: {data.id}, Name: {data.name}, Optional: {data.optional || 'N/A'}
            </div>
          )}
        </AsyncDataWrapper>
      );

      expect(screen.getByText('ID: 1, Name: Typed Item, Optional: N/A')).toBeInTheDocument();
    });
  });
});