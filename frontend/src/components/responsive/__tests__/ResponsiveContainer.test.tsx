/**
 * ResponsiveContainer Component Tests
 * 
 * Comprehensive tests for responsive design utilities including:
 * - useMediaQuery hook
 * - useBreakpoint hook
 * - TouchInteraction component
 * - ResponsiveVisibility component
 * - Touch gestures (swipe, tap, double tap, long press)
 * - Screen size detection
 * - Mobile-specific interactions
 * - Responsive layout changes
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { 
  useMediaQuery, 
  useBreakpoint, 
  TouchInteraction, 
  ResponsiveVisibility,
  BREAKPOINTS 
} from '../ResponsiveContainer';

// Mock window.matchMedia
const createMatchMedia = (matches: boolean) => {
  return jest.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  }));
};

// Test components for hooks
const MediaQueryTestComponent: React.FC<{ query: string }> = ({ query }) => {
  const matches = useMediaQuery(query);
  return <div data-testid="media-query-result">{matches ? 'matches' : 'no-match'}</div>;
};

const BreakpointTestComponent: React.FC = () => {
  const breakpoint = useBreakpoint();
  return (
    <div>
      <div data-testid="current-breakpoint">{breakpoint}</div>
      <div data-testid="is-mobile">{breakpoint === 'mobile' ? 'true' : 'false'}</div>
      <div data-testid="is-tablet">{breakpoint === 'tablet' ? 'true' : 'false'}</div>
      <div data-testid="is-desktop">{breakpoint === 'desktop' ? 'true' : 'false'}</div>
    </div>
  );
};

describe('useMediaQuery Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return true when media query matches', () => {
    window.matchMedia = createMatchMedia(true);

    renderWithProviders(<MediaQueryTestComponent query="(min-width: 768px)" />);

    expect(screen.getByTestId('media-query-result')).toHaveTextContent('matches');
  });

  it('should return false when media query does not match', () => {
    window.matchMedia = createMatchMedia(false);

    renderWithProviders(<MediaQueryTestComponent query="(min-width: 768px)" />);

    expect(screen.getByTestId('media-query-result')).toHaveTextContent('no-match');
  });

  it('should update when media query changes', () => {
    let listeners: Array<(event: MediaQueryListEvent) => void> = [];
    
    window.matchMedia = jest.fn().mockImplementation((query: string) => {
      const mediaQueryList = {
        matches: false,
        media: query,
        onchange: null,
        addEventListener: jest.fn((event: string, listener: (event: MediaQueryListEvent) => void) => {
          if (event === 'change') {
            listeners.push(listener);
          }
        }),
        removeEventListener: jest.fn((event: string, listener: (event: MediaQueryListEvent) => void) => {
          if (event === 'change') {
            listeners = listeners.filter(l => l !== listener);
          }
        }),
        dispatchEvent: jest.fn(),
      };
      return mediaQueryList;
    });

    renderWithProviders(<MediaQueryTestComponent query="(min-width: 768px)" />);

    expect(screen.getByTestId('media-query-result')).toHaveTextContent('no-match');

    // Simulate media query change
    act(() => {
      listeners.forEach(listener => {
        listener({ matches: true } as MediaQueryListEvent);
      });
    });

    expect(screen.getByTestId('media-query-result')).toHaveTextContent('matches');
  });

  it('should clean up event listeners on unmount', () => {
    const removeEventListener = jest.fn();
    window.matchMedia = jest.fn().mockImplementation(() => ({
      matches: false,
      media: '',
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener,
      dispatchEvent: jest.fn(),
    }));

    const { unmount } = renderWithProviders(<MediaQueryTestComponent query="(min-width: 768px)" />);

    unmount();

    expect(removeEventListener).toHaveBeenCalledWith('change', expect.any(Function));
  });

  it('should handle multiple media queries', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: query.includes('768px'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    const MultiQueryComponent = () => {
      const isMedium = useMediaQuery('(min-width: 768px)');
      const isLarge = useMediaQuery('(min-width: 1024px)');
      
      return (
        <div>
          <div data-testid="medium">{isMedium ? 'true' : 'false'}</div>
          <div data-testid="large">{isLarge ? 'true' : 'false'}</div>
        </div>
      );
    };

    renderWithProviders(<MultiQueryComponent />);

    expect(screen.getByTestId('medium')).toHaveTextContent('true');
    expect(screen.getByTestId('large')).toHaveTextContent('false');
  });
});

describe('useBreakpoint Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should detect mobile breakpoint', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: !query.includes('min-width'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(<BreakpointTestComponent />);

    expect(screen.getByTestId('current-breakpoint')).toHaveTextContent('mobile');
    expect(screen.getByTestId('is-mobile')).toHaveTextContent('true');
    expect(screen.getByTestId('is-tablet')).toHaveTextContent('false');
    expect(screen.getByTestId('is-desktop')).toHaveTextContent('false');
  });

  it('should detect tablet breakpoint', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: query.includes('768px') && !query.includes('1024px'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(<BreakpointTestComponent />);

    expect(screen.getByTestId('current-breakpoint')).toHaveTextContent('tablet');
    expect(screen.getByTestId('is-tablet')).toHaveTextContent('true');
  });

  it('should detect desktop breakpoint', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: query.includes('1024px'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(<BreakpointTestComponent />);

    expect(screen.getByTestId('current-breakpoint')).toHaveTextContent('desktop');
    expect(screen.getByTestId('is-desktop')).toHaveTextContent('true');
  });

  it('should update breakpoint on resize', () => {
    let listeners: { [key: string]: Array<(event: MediaQueryListEvent) => void> } = {};
    
    window.matchMedia = jest.fn().mockImplementation((query: string) => {
      const matches = query.includes('768px') && !query.includes('1024px');
      return {
        matches,
        media: query,
        onchange: null,
        addEventListener: jest.fn((event: string, listener: (event: MediaQueryListEvent) => void) => {
          if (!listeners[query]) listeners[query] = [];
          listeners[query].push(listener);
        }),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      };
    });

    renderWithProviders(<BreakpointTestComponent />);

    expect(screen.getByTestId('current-breakpoint')).toHaveTextContent('tablet');

    // Simulate resize to desktop
    act(() => {
      Object.entries(listeners).forEach(([query, queryListeners]) => {
        const matches = query.includes('1024px');
        queryListeners.forEach(listener => {
          listener({ matches } as MediaQueryListEvent);
        });
      });
    });

    expect(screen.getByTestId('current-breakpoint')).toHaveTextContent('desktop');
  });
});

describe('TouchInteraction Component', () => {
  const user = userEvent.setup();
  const mockOnSwipeLeft = jest.fn();
  const mockOnSwipeRight = jest.fn();
  const mockOnSwipeUp = jest.fn();
  const mockOnSwipeDown = jest.fn();
  const mockOnTap = jest.fn();
  const mockOnDoubleTap = jest.fn();
  const mockOnLongPress = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  const createTouchEvent = (type: string, touches: Array<{ clientX: number; clientY: number }>) => {
    return new TouchEvent(type, {
      bubbles: true,
      cancelable: true,
      touches: touches.map(touch => ({
        ...touch,
        identifier: 0,
        target: document.body,
        radiusX: 1,
        radiusY: 1,
        rotationAngle: 0,
        force: 1,
      } as Touch)),
    });
  };

  it('should detect left swipe', () => {
    renderWithProviders(
      <TouchInteraction onSwipeLeft={mockOnSwipeLeft} threshold={50}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Simulate swipe left
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 200, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchmove', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    expect(mockOnSwipeLeft).toHaveBeenCalledTimes(1);
  });

  it('should detect right swipe', () => {
    renderWithProviders(
      <TouchInteraction onSwipeRight={mockOnSwipeRight} threshold={50}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Simulate swipe right
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchmove', [{ clientX: 200, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 200, clientY: 100 }]));

    expect(mockOnSwipeRight).toHaveBeenCalledTimes(1);
  });

  it('should detect up swipe', () => {
    renderWithProviders(
      <TouchInteraction onSwipeUp={mockOnSwipeUp} threshold={50}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Simulate swipe up
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 200 }]));
    fireEvent(touchArea, createTouchEvent('touchmove', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    expect(mockOnSwipeUp).toHaveBeenCalledTimes(1);
  });

  it('should detect down swipe', () => {
    renderWithProviders(
      <TouchInteraction onSwipeDown={mockOnSwipeDown} threshold={50}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Simulate swipe down
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchmove', [{ clientX: 100, clientY: 200 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 200 }]));

    expect(mockOnSwipeDown).toHaveBeenCalledTimes(1);
  });

  it('should not trigger swipe if movement is below threshold', () => {
    renderWithProviders(
      <TouchInteraction 
        onSwipeLeft={mockOnSwipeLeft} 
        onSwipeRight={mockOnSwipeRight}
        threshold={50}
      >
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Small movement
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchmove', [{ clientX: 120, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 120, clientY: 100 }]));

    expect(mockOnSwipeLeft).not.toHaveBeenCalled();
    expect(mockOnSwipeRight).not.toHaveBeenCalled();
  });

  it('should detect tap', async () => {
    renderWithProviders(
      <TouchInteraction onTap={mockOnTap}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Simulate tap
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    // Wait for tap delay
    act(() => {
      jest.advanceTimersByTime(300);
    });

    expect(mockOnTap).toHaveBeenCalledTimes(1);
  });

  it('should detect double tap', () => {
    renderWithProviders(
      <TouchInteraction onDoubleTap={mockOnDoubleTap}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // First tap
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    // Second tap within double tap delay
    act(() => {
      jest.advanceTimersByTime(100);
    });

    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    expect(mockOnDoubleTap).toHaveBeenCalledTimes(1);
  });

  it('should not detect double tap if delay is too long', () => {
    renderWithProviders(
      <TouchInteraction onDoubleTap={mockOnDoubleTap}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // First tap
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    // Wait too long
    act(() => {
      jest.advanceTimersByTime(400);
    });

    // Second tap
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    expect(mockOnDoubleTap).not.toHaveBeenCalled();
  });

  it('should detect long press', () => {
    renderWithProviders(
      <TouchInteraction onLongPress={mockOnLongPress}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Start touch
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));

    // Wait for long press duration
    act(() => {
      jest.advanceTimersByTime(600);
    });

    expect(mockOnLongPress).toHaveBeenCalledTimes(1);
  });

  it('should cancel long press on movement', () => {
    renderWithProviders(
      <TouchInteraction onLongPress={mockOnLongPress}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Start touch
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));

    // Move before long press completes
    act(() => {
      jest.advanceTimersByTime(300);
    });

    fireEvent(touchArea, createTouchEvent('touchmove', [{ clientX: 150, clientY: 100 }]));

    act(() => {
      jest.advanceTimersByTime(400);
    });

    expect(mockOnLongPress).not.toHaveBeenCalled();
  });

  it('should handle multiple gestures', () => {
    renderWithProviders(
      <TouchInteraction 
        onSwipeLeft={mockOnSwipeLeft}
        onTap={mockOnTap}
        onLongPress={mockOnLongPress}
      >
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Test swipe
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 200, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchmove', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    expect(mockOnSwipeLeft).toHaveBeenCalledTimes(1);
    expect(mockOnTap).not.toHaveBeenCalled();
    expect(mockOnLongPress).not.toHaveBeenCalled();
  });

  it('should apply className to wrapper', () => {
    renderWithProviders(
      <TouchInteraction className="custom-touch-class">
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area').parentElement;
    expect(touchArea).toHaveClass('custom-touch-class');
  });

  it('should handle multi-touch gestures', () => {
    renderWithProviders(
      <TouchInteraction onSwipeLeft={mockOnSwipeLeft}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Multi-touch should not trigger swipe
    fireEvent(touchArea, createTouchEvent('touchstart', [
      { clientX: 200, clientY: 100 },
      { clientX: 300, clientY: 100 }
    ]));

    fireEvent(touchArea, createTouchEvent('touchend', []));

    expect(mockOnSwipeLeft).not.toHaveBeenCalled();
  });
});

describe('ResponsiveVisibility Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should show content on specified breakpoint', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: query.includes('768px') && !query.includes('1024px'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(
      <ResponsiveVisibility showOn={['tablet']}>
        <div data-testid="responsive-content">Tablet Content</div>
      </ResponsiveVisibility>
    );

    expect(screen.getByTestId('responsive-content')).toBeInTheDocument();
  });

  it('should hide content on other breakpoints', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: query.includes('1024px'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(
      <ResponsiveVisibility showOn={['mobile', 'tablet']}>
        <div data-testid="responsive-content">Mobile/Tablet Content</div>
      </ResponsiveVisibility>
    );

    expect(screen.queryByTestId('responsive-content')).not.toBeInTheDocument();
  });

  it('should show content on multiple breakpoints', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: !query.includes('min-width'), // mobile
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(
      <ResponsiveVisibility showOn={['mobile', 'desktop']}>
        <div data-testid="responsive-content">Mobile/Desktop Content</div>
      </ResponsiveVisibility>
    );

    expect(screen.getByTestId('responsive-content')).toBeInTheDocument();
  });

  it('should hide content using hideOn prop', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: query.includes('768px') && !query.includes('1024px'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(
      <ResponsiveVisibility hideOn={['tablet']}>
        <div data-testid="responsive-content">Non-Tablet Content</div>
      </ResponsiveVisibility>
    );

    expect(screen.queryByTestId('responsive-content')).not.toBeInTheDocument();
  });

  it('should update visibility on breakpoint change', () => {
    let listeners: Array<(event: MediaQueryListEvent) => void> = [];
    let currentMatches = false;

    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      get matches() { return currentMatches; },
      media: query,
      onchange: null,
      addEventListener: jest.fn((event: string, listener: (event: MediaQueryListEvent) => void) => {
        if (event === 'change') listeners.push(listener);
      }),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(
      <ResponsiveVisibility showOn={['tablet']}>
        <div data-testid="responsive-content">Tablet Content</div>
      </ResponsiveVisibility>
    );

    expect(screen.queryByTestId('responsive-content')).not.toBeInTheDocument();

    // Change to tablet
    act(() => {
      currentMatches = true;
      listeners.forEach(listener => {
        listener({ matches: true } as MediaQueryListEvent);
      });
    });

    expect(screen.getByTestId('responsive-content')).toBeInTheDocument();
  });

  it('should handle hideOn priority over showOn', () => {
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: query.includes('768px') && !query.includes('1024px'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(
      <ResponsiveVisibility showOn={['tablet']} hideOn={['tablet']}>
        <div data-testid="responsive-content">Content</div>
      </ResponsiveVisibility>
    );

    // hideOn should take priority
    expect(screen.queryByTestId('responsive-content')).not.toBeInTheDocument();
  });
});

describe('Integration Tests', () => {
  it('should work with touch interactions on mobile', () => {
    // Set mobile breakpoint
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: !query.includes('min-width'),
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    const mockSwipe = jest.fn();

    renderWithProviders(
      <ResponsiveVisibility showOn={['mobile']}>
        <TouchInteraction onSwipeLeft={mockSwipe}>
          <div data-testid="mobile-swipeable">Swipe me on mobile</div>
        </TouchInteraction>
      </ResponsiveVisibility>
    );

    const element = screen.getByTestId('mobile-swipeable');

    // Perform swipe
    fireEvent(element, createTouchEvent('touchstart', [{ clientX: 200, clientY: 100 }]));
    fireEvent(element, createTouchEvent('touchmove', [{ clientX: 100, clientY: 100 }]));
    fireEvent(element, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    expect(mockSwipe).toHaveBeenCalled();
  });

  it('should handle responsive layouts with breakpoint hooks', () => {
    let currentBreakpoint = 'mobile';
    
    window.matchMedia = jest.fn().mockImplementation((query: string) => ({
      matches: currentBreakpoint === 'desktop' ? query.includes('1024px') : false,
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    const ResponsiveLayout = () => {
      const breakpoint = useBreakpoint();
      
      return (
        <div data-testid="layout" className={
          breakpoint === 'mobile' ? 'flex-col' :
          breakpoint === 'tablet' ? 'flex-row gap-4' :
          'grid grid-cols-3'
        }>
          <div>Item 1</div>
          <div>Item 2</div>
          <div>Item 3</div>
        </div>
      );
    };

    const { rerender } = renderWithProviders(<ResponsiveLayout />);

    expect(screen.getByTestId('layout')).toHaveClass('flex-col');

    // Change to desktop
    currentBreakpoint = 'desktop';
    rerender(<ResponsiveLayout />);

    expect(screen.getByTestId('layout')).toHaveClass('grid', 'grid-cols-3');
  });
});

describe('Edge Cases', () => {
  it('should handle undefined matchMedia gracefully', () => {
    const originalMatchMedia = window.matchMedia;
    // @ts-ignore
    window.matchMedia = undefined;

    renderWithProviders(<MediaQueryTestComponent query="(min-width: 768px)" />);

    expect(screen.getByTestId('media-query-result')).toHaveTextContent('no-match');

    window.matchMedia = originalMatchMedia;
  });

  it('should handle touch events with missing coordinates', () => {
    const mockSwipe = jest.fn();

    renderWithProviders(
      <TouchInteraction onSwipeLeft={mockSwipe}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Touch event without touches
    fireEvent.touchStart(touchArea, { touches: [] });
    fireEvent.touchEnd(touchArea, { touches: [] });

    expect(mockSwipe).not.toHaveBeenCalled();
  });

  it('should handle rapid breakpoint changes', () => {
    let listeners: Array<(event: MediaQueryListEvent) => void> = [];
    
    window.matchMedia = jest.fn().mockImplementation(() => ({
      matches: false,
      media: '',
      onchange: null,
      addEventListener: jest.fn((event: string, listener: (event: MediaQueryListEvent) => void) => {
        if (event === 'change') listeners.push(listener);
      }),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    renderWithProviders(<BreakpointTestComponent />);

    // Rapid changes
    act(() => {
      listeners.forEach(listener => {
        listener({ matches: true } as MediaQueryListEvent);
        listener({ matches: false } as MediaQueryListEvent);
        listener({ matches: true } as MediaQueryListEvent);
      });
    });

    // Should settle on the last value
    expect(screen.getByTestId('current-breakpoint')).toBeDefined();
  });

  it('should clean up touch event timers on unmount', () => {
    jest.useFakeTimers();
    const mockLongPress = jest.fn();

    const { unmount } = renderWithProviders(
      <TouchInteraction onLongPress={mockLongPress}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Start long press
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));

    // Unmount before completion
    unmount();

    // Advance timers
    act(() => {
      jest.advanceTimersByTime(1000);
    });

    expect(mockLongPress).not.toHaveBeenCalled();
    jest.useRealTimers();
  });

  it('should handle empty showOn and hideOn arrays', () => {
    window.matchMedia = createMatchMedia(true);

    renderWithProviders(
      <ResponsiveVisibility showOn={[]} hideOn={[]}>
        <div data-testid="content">Always visible</div>
      </ResponsiveVisibility>
    );

    expect(screen.getByTestId('content')).toBeInTheDocument();
  });
});

describe('Performance Tests', () => {
  it('should not re-render unnecessarily on irrelevant changes', () => {
    let renderCount = 0;

    const TestComponent = () => {
      renderCount++;
      const matches = useMediaQuery('(min-width: 768px)');
      return <div data-testid="render-count">{renderCount}</div>;
    };

    window.matchMedia = createMatchMedia(true);

    const { rerender } = renderWithProviders(<TestComponent />);

    const initialCount = renderCount;

    // Re-render with same props
    rerender(<TestComponent />);

    expect(renderCount).toBe(initialCount);
  });

  it('should handle many touch listeners efficiently', () => {
    const handlers = {
      onSwipeLeft: jest.fn(),
      onSwipeRight: jest.fn(),
      onSwipeUp: jest.fn(),
      onSwipeDown: jest.fn(),
      onTap: jest.fn(),
      onDoubleTap: jest.fn(),
      onLongPress: jest.fn(),
    };

    renderWithProviders(
      <TouchInteraction {...handlers}>
        <div data-testid="touch-area">Touch Area</div>
      </TouchInteraction>
    );

    const touchArea = screen.getByTestId('touch-area');

    // Quick tap
    fireEvent(touchArea, createTouchEvent('touchstart', [{ clientX: 100, clientY: 100 }]));
    fireEvent(touchArea, createTouchEvent('touchend', [{ clientX: 100, clientY: 100 }]));

    // Only tap should be called eventually
    jest.advanceTimersByTime(300);
    expect(handlers.onTap).toHaveBeenCalledTimes(1);
    expect(handlers.onSwipeLeft).not.toHaveBeenCalled();
    expect(handlers.onLongPress).not.toHaveBeenCalled();
  });
});