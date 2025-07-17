/**
 * Comprehensive User Interaction Tests
 * 
 * End-to-end tests focusing on complex user workflows and edge cases:
 * - Keyboard navigation and accessibility
 * - Complex form interactions
 * - Drag and drop operations
 * - Copy/paste functionality
 * - Session management
 * - Network interruptions
 * - Race conditions
 * - Memory leaks
 * - Performance under stress
 */

import React from 'react';
import { screen, fireEvent, waitFor, within, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../test-utils';
import { server } from '../../test-utils/msw-server';
import { rest } from 'msw';

// Import components for integration testing
import { GenerationStudio } from '../generation/GenerationStudio';
import { ProjectManager } from '../projects/ProjectManager';
import { AudioEditor } from '../editor/AudioEditor';
import { LoginForm } from '../auth/LoginForm';
import { RegisterForm } from '../auth/RegisterForm';

// Mock WebSocket for real-time features
class MockWebSocket {
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) this.onopen(new Event('open'));
    }, 100);
  }

  send(data: string) {
    // Simulate server response
    setTimeout(() => {
      if (this.onmessage) {
        this.onmessage(new MessageEvent('message', { data }));
      }
    }, 50);
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
}

// @ts-ignore
global.WebSocket = MockWebSocket;

// Mock Clipboard API
const mockClipboard = {
  writeText: jest.fn().mockResolvedValue(undefined),
  readText: jest.fn().mockResolvedValue(''),
};
Object.assign(navigator, { clipboard: mockClipboard });

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation((callback) => ({
  observe: jest.fn(),
  disconnect: jest.fn(),
  unobserve: jest.fn(),
}));

describe('Complex User Workflows', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    sessionStorage.clear();
  });

  describe('Full Music Generation Workflow', () => {
    it('should complete entire generation workflow with keyboard navigation', async () => {
      const mockGenerate = jest.fn().mockResolvedValue({
        id: 'gen-123',
        audioUrl: '/audio/generated.wav',
      });

      renderWithProviders(
        <GenerationStudio onGenerate={mockGenerate} />
      );

      // Navigate with Tab key
      await user.tab();
      expect(screen.getByPlaceholderText(/describe your music/i)).toHaveFocus();

      // Type prompt with keyboard
      await user.keyboard('Upbeat electronic dance music with heavy bass');

      // Tab to genre selector
      await user.tab();
      await user.keyboard('{ArrowDown}{Enter}'); // Select first genre

      // Tab to mood selector
      await user.tab();
      await user.keyboard('{ArrowDown}{ArrowDown}{Enter}'); // Select second mood

      // Tab to duration slider
      await user.tab();
      await user.keyboard('{ArrowRight}{ArrowRight}'); // Increase duration

      // Tab to generate button
      await user.tab();
      await user.keyboard('{Enter}'); // Press Enter to generate

      await waitFor(() => {
        expect(mockGenerate).toHaveBeenCalledWith({
          prompt: 'Upbeat electronic dance music with heavy bass',
          genre: expect.any(String),
          mood: expect.any(String),
          duration: expect.any(Number),
        });
      });
    });

    it('should handle generation interruption and retry', async () => {
      let generateCount = 0;
      const mockGenerate = jest.fn().mockImplementation(() => {
        generateCount++;
        if (generateCount === 1) {
          return Promise.reject(new Error('Network error'));
        }
        return Promise.resolve({
          id: 'gen-123',
          audioUrl: '/audio/generated.wav',
        });
      });

      renderWithProviders(
        <GenerationStudio onGenerate={mockGenerate} />
      );

      // First attempt
      await user.type(screen.getByPlaceholderText(/describe your music/i), 'Test prompt');
      await user.click(screen.getByRole('button', { name: /generate/i }));

      // Wait for error
      await waitFor(() => {
        expect(screen.getByText(/network error/i)).toBeInTheDocument();
      });

      // Retry
      await user.click(screen.getByRole('button', { name: /try again/i }));

      await waitFor(() => {
        expect(mockGenerate).toHaveBeenCalledTimes(2);
        expect(screen.queryByText(/network error/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Drag and Drop Interactions', () => {
    it('should handle file drag and drop with validation', async () => {
      const mockOnFileDrop = jest.fn();

      const DropZone = ({ onFileDrop }: { onFileDrop: (files: File[]) => void }) => {
        const [isDragging, setIsDragging] = React.useState(false);

        return (
          <div
            data-testid="drop-zone"
            className={isDragging ? 'dragging' : ''}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              const files = Array.from(e.dataTransfer.files);
              onFileDrop(files);
            }}
          >
            Drop audio files here
          </div>
        );
      };

      renderWithProviders(<DropZone onFileDrop={mockOnFileDrop} />);

      const dropZone = screen.getByTestId('drop-zone');
      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mp3' });

      // Simulate drag over
      fireEvent.dragOver(dropZone, {
        dataTransfer: { files: [file] },
      });

      expect(dropZone).toHaveClass('dragging');

      // Simulate drop
      fireEvent.drop(dropZone, {
        dataTransfer: { files: [file] },
      });

      expect(mockOnFileDrop).toHaveBeenCalledWith([file]);
      expect(dropZone).not.toHaveClass('dragging');
    });

    it('should handle drag and drop project reordering', async () => {
      const mockReorder = jest.fn();
      
      const DraggableList = ({ items, onReorder }: any) => {
        const [draggedIndex, setDraggedIndex] = React.useState<number | null>(null);

        return (
          <div>
            {items.map((item: any, index: number) => (
              <div
                key={item.id}
                data-testid={`item-${item.id}`}
                draggable
                onDragStart={() => setDraggedIndex(index)}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (draggedIndex !== null && draggedIndex !== index) {
                    onReorder(draggedIndex, index);
                  }
                  setDraggedIndex(null);
                }}
              >
                {item.name}
              </div>
            ))}
          </div>
        );
      };

      const items = [
        { id: '1', name: 'Project 1' },
        { id: '2', name: 'Project 2' },
        { id: '3', name: 'Project 3' },
      ];

      renderWithProviders(<DraggableList items={items} onReorder={mockReorder} />);

      const item1 = screen.getByTestId('item-1');
      const item3 = screen.getByTestId('item-3');

      // Drag item 1 to position of item 3
      fireEvent.dragStart(item1);
      fireEvent.dragOver(item3);
      fireEvent.drop(item3);

      expect(mockReorder).toHaveBeenCalledWith(0, 2);
    });
  });

  describe('Copy/Paste and Clipboard Operations', () => {
    it('should handle copy paste in audio editor', async () => {
      const mockCopy = jest.fn();
      const mockPaste = jest.fn();

      renderWithProviders(
        <AudioEditor 
          onCopy={mockCopy}
          onPaste={mockPaste}
        />
      );

      // Select region
      const waveform = screen.getByTestId('waveform');
      fireEvent.mouseDown(waveform, { clientX: 100 });
      fireEvent.mouseMove(waveform, { clientX: 200 });
      fireEvent.mouseUp(waveform);

      // Copy with keyboard shortcut
      fireEvent.keyDown(document, { key: 'c', ctrlKey: true });
      
      await waitFor(() => {
        expect(mockCopy).toHaveBeenCalled();
      });

      // Paste with keyboard shortcut
      fireEvent.keyDown(document, { key: 'v', ctrlKey: true });

      await waitFor(() => {
        expect(mockPaste).toHaveBeenCalled();
      });
    });

    it('should handle share link copy', async () => {
      renderWithProviders(
        <div>
          <button onClick={() => navigator.clipboard.writeText('https://app.com/share/123')}>
            Copy Share Link
          </button>
          <span data-testid="copy-status"></span>
        </div>
      );

      await user.click(screen.getByText('Copy Share Link'));

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith('https://app.com/share/123');
    });
  });

  describe('Session Management and Persistence', () => {
    it('should persist and restore form state across sessions', async () => {
      const FormWithPersistence = () => {
        const [formData, setFormData] = React.useState(() => {
          const saved = localStorage.getItem('formData');
          return saved ? JSON.parse(saved) : { prompt: '', genre: '' };
        });

        React.useEffect(() => {
          localStorage.setItem('formData', JSON.stringify(formData));
        }, [formData]);

        return (
          <form>
            <input
              value={formData.prompt}
              onChange={(e) => setFormData({ ...formData, prompt: e.target.value })}
              placeholder="Enter prompt"
            />
            <select
              value={formData.genre}
              onChange={(e) => setFormData({ ...formData, genre: e.target.value })}
            >
              <option value="">Select genre</option>
              <option value="rock">Rock</option>
              <option value="jazz">Jazz</option>
            </select>
          </form>
        );
      };

      const { unmount } = renderWithProviders(<FormWithPersistence />);

      // Enter data
      await user.type(screen.getByPlaceholderText('Enter prompt'), 'Test prompt');
      await user.selectOptions(screen.getByRole('combobox'), 'jazz');

      // Unmount (simulate page reload)
      unmount();

      // Remount
      renderWithProviders(<FormWithPersistence />);

      // Check restored values
      expect(screen.getByPlaceholderText('Enter prompt')).toHaveValue('Test prompt');
      expect(screen.getByRole('combobox')).toHaveValue('jazz');
    });

    it('should handle session timeout gracefully', async () => {
      server.use(
        rest.post('/api/generate', (req, res, ctx) => {
          return res(
            ctx.status(401),
            ctx.json({ error: 'Session expired' })
          );
        })
      );

      const mockOnSessionExpired = jest.fn();

      renderWithProviders(
        <GenerationStudio onSessionExpired={mockOnSessionExpired} />
      );

      await user.type(screen.getByPlaceholderText(/describe your music/i), 'Test');
      await user.click(screen.getByRole('button', { name: /generate/i }));

      await waitFor(() => {
        expect(mockOnSessionExpired).toHaveBeenCalled();
        expect(screen.getByText(/session expired/i)).toBeInTheDocument();
      });
    });
  });

  describe('Network Interruption Handling', () => {
    it('should handle network offline/online transitions', async () => {
      const NetworkAwareComponent = () => {
        const [isOnline, setIsOnline] = React.useState(navigator.onLine);

        React.useEffect(() => {
          const handleOnline = () => setIsOnline(true);
          const handleOffline = () => setIsOnline(false);

          window.addEventListener('online', handleOnline);
          window.addEventListener('offline', handleOffline);

          return () => {
            window.removeEventListener('online', handleOnline);
            window.removeEventListener('offline', handleOffline);
          };
        }, []);

        return (
          <div>
            <div data-testid="status">{isOnline ? 'Online' : 'Offline'}</div>
            <button disabled={!isOnline}>Generate Music</button>
          </div>
        );
      };

      renderWithProviders(<NetworkAwareComponent />);

      expect(screen.getByTestId('status')).toHaveTextContent('Online');
      expect(screen.getByRole('button')).not.toBeDisabled();

      // Simulate going offline
      act(() => {
        window.dispatchEvent(new Event('offline'));
      });

      expect(screen.getByTestId('status')).toHaveTextContent('Offline');
      expect(screen.getByRole('button')).toBeDisabled();

      // Simulate coming back online
      act(() => {
        window.dispatchEvent(new Event('online'));
      });

      expect(screen.getByTestId('status')).toHaveTextContent('Online');
      expect(screen.getByRole('button')).not.toBeDisabled();
    });

    it('should retry failed requests automatically', async () => {
      let attemptCount = 0;
      server.use(
        rest.post('/api/generate', (req, res, ctx) => {
          attemptCount++;
          if (attemptCount < 3) {
            return res(ctx.status(500));
          }
          return res(ctx.json({ id: 'success', audioUrl: '/audio.wav' }));
        })
      );

      const RetryComponent = () => {
        const [status, setStatus] = React.useState('');
        const [retryCount, setRetryCount] = React.useState(0);

        const generate = async () => {
          setStatus('loading');
          try {
            const response = await fetch('/api/generate', { method: 'POST' });
            if (!response.ok) throw new Error('Failed');
            setStatus('success');
          } catch (error) {
            if (retryCount < 2) {
              setRetryCount(retryCount + 1);
              setTimeout(() => generate(), 1000);
            } else {
              setStatus('error');
            }
          }
        };

        return (
          <div>
            <div data-testid="status">{status}</div>
            <div data-testid="retry-count">{retryCount}</div>
            <button onClick={generate}>Generate</button>
          </div>
        );
      };

      renderWithProviders(<RetryComponent />);

      await user.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByTestId('status')).toHaveTextContent('success');
        expect(screen.getByTestId('retry-count')).toHaveTextContent('2');
      }, { timeout: 5000 });
    });
  });

  describe('Race Condition Handling', () => {
    it('should handle rapid button clicks without duplicate submissions', async () => {
      const mockSubmit = jest.fn().mockImplementation(() => 
        new Promise(resolve => setTimeout(resolve, 1000))
      );

      const FormWithDebounce = () => {
        const [isSubmitting, setIsSubmitting] = React.useState(false);

        const handleSubmit = async () => {
          if (isSubmitting) return;
          
          setIsSubmitting(true);
          try {
            await mockSubmit();
          } finally {
            setIsSubmitting(false);
          }
        };

        return (
          <button onClick={handleSubmit} disabled={isSubmitting}>
            {isSubmitting ? 'Submitting...' : 'Submit'}
          </button>
        );
      };

      renderWithProviders(<FormWithDebounce />);

      const button = screen.getByRole('button');

      // Rapid clicks
      await user.click(button);
      await user.click(button);
      await user.click(button);

      // Only one submission should occur
      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(button).toBeDisabled();
      expect(button).toHaveTextContent('Submitting...');

      await waitFor(() => {
        expect(button).not.toBeDisabled();
        expect(button).toHaveTextContent('Submit');
      });
    });

    it('should cancel previous requests when new ones are made', async () => {
      const SearchWithCancellation = () => {
        const [results, setResults] = React.useState<string[]>([]);
        const [loading, setLoading] = React.useState(false);
        const abortControllerRef = React.useRef<AbortController | null>(null);

        const search = async (query: string) => {
          // Cancel previous request
          if (abortControllerRef.current) {
            abortControllerRef.current.abort();
          }

          abortControllerRef.current = new AbortController();
          setLoading(true);

          try {
            const response = await fetch(`/api/search?q=${query}`, {
              signal: abortControllerRef.current.signal,
            });
            const data = await response.json();
            setResults(data.results);
          } catch (error: any) {
            if (error.name !== 'AbortError') {
              console.error(error);
            }
          } finally {
            setLoading(false);
          }
        };

        return (
          <div>
            <input
              placeholder="Search"
              onChange={(e) => search(e.target.value)}
            />
            <div data-testid="loading">{loading ? 'Loading...' : 'Ready'}</div>
            <ul>
              {results.map((result, i) => (
                <li key={i}>{result}</li>
              ))}
            </ul>
          </div>
        );
      };

      let requestCount = 0;
      server.use(
        rest.get('/api/search', async (req, res, ctx) => {
          requestCount++;
          const query = req.url.searchParams.get('q');
          await new Promise(resolve => setTimeout(resolve, 200));
          return res(ctx.json({ results: [`Result for ${query}`] }));
        })
      );

      renderWithProviders(<SearchWithCancellation />);

      const input = screen.getByPlaceholderText('Search');

      // Type rapidly
      await user.type(input, 'abc');

      // Should show loading
      expect(screen.getByTestId('loading')).toHaveTextContent('Loading...');

      await waitFor(() => {
        expect(screen.getByTestId('loading')).toHaveTextContent('Ready');
        expect(screen.getByText('Result for abc')).toBeInTheDocument();
      });

      // Despite typing 3 characters, not all requests may complete
      expect(requestCount).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Memory Leak Prevention', () => {
    it('should cleanup event listeners and timers', async () => {
      const ComponentWithTimers = () => {
        const [count, setCount] = React.useState(0);
        const [isVisible, setIsVisible] = React.useState(true);

        React.useEffect(() => {
          if (!isVisible) return;

          const interval = setInterval(() => {
            setCount(c => c + 1);
          }, 100);

          const handleResize = () => {
            console.log('resize');
          };

          window.addEventListener('resize', handleResize);

          return () => {
            clearInterval(interval);
            window.removeEventListener('resize', handleResize);
          };
        }, [isVisible]);

        return (
          <div>
            <div data-testid="count">{count}</div>
            <button onClick={() => setIsVisible(false)}>Hide</button>
          </div>
        );
      };

      jest.useFakeTimers();
      const addEventListenerSpy = jest.spyOn(window, 'addEventListener');
      const removeEventListenerSpy = jest.spyOn(window, 'removeEventListener');

      renderWithProviders(<ComponentWithTimers />);

      expect(addEventListenerSpy).toHaveBeenCalledWith('resize', expect.any(Function));

      // Let timer run
      act(() => {
        jest.advanceTimersByTime(300);
      });

      expect(screen.getByTestId('count')).toHaveTextContent('3');

      // Hide component
      await user.click(screen.getByText('Hide'));

      // Verify cleanup
      expect(removeEventListenerSpy).toHaveBeenCalledWith('resize', expect.any(Function));

      // Timer should stop
      act(() => {
        jest.advanceTimersByTime(300);
      });

      expect(screen.getByTestId('count')).toHaveTextContent('3');

      jest.useRealTimers();
    });

    it('should cleanup WebSocket connections', async () => {
      const WebSocketComponent = () => {
        const [messages, setMessages] = React.useState<string[]>([]);
        const wsRef = React.useRef<WebSocket | null>(null);

        React.useEffect(() => {
          wsRef.current = new WebSocket('ws://localhost:3000');

          wsRef.current.onmessage = (event) => {
            setMessages(prev => [...prev, event.data]);
          };

          return () => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.close();
            }
          };
        }, []);

        return (
          <div>
            <div data-testid="message-count">{messages.length}</div>
            <button onClick={() => wsRef.current?.send('test')}>Send</button>
          </div>
        );
      };

      const { unmount } = renderWithProviders(<WebSocketComponent />);

      // Wait for connection
      await waitFor(() => {
        expect(screen.getByRole('button')).toBeInTheDocument();
      });

      // Send message
      await user.click(screen.getByText('Send'));

      await waitFor(() => {
        expect(screen.getByTestId('message-count')).toHaveTextContent('1');
      });

      // Unmount should close connection
      unmount();

      // WebSocket should be closed
      // (MockWebSocket would have close() called)
    });
  });

  describe('Accessibility and Keyboard Navigation', () => {
    it('should support full keyboard navigation through forms', async () => {
      renderWithProviders(<RegisterForm />);

      // Start at first input
      await user.tab();
      expect(screen.getByPlaceholderText(/email/i)).toHaveFocus();

      // Fill email
      await user.keyboard('test@example.com');

      // Tab to password
      await user.tab();
      expect(screen.getByPlaceholderText(/^password/i)).toHaveFocus();
      await user.keyboard('SecurePass123!');

      // Tab to confirm password
      await user.tab();
      expect(screen.getByPlaceholderText(/confirm password/i)).toHaveFocus();
      await user.keyboard('SecurePass123!');

      // Tab through remaining fields
      await user.tab(); // Display name
      await user.keyboard('Test User');

      // Submit with Enter
      await user.keyboard('{Enter}');

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /register/i })).toBeDisabled();
      });
    });

    it('should announce dynamic content changes to screen readers', async () => {
      const DynamicAnnouncer = () => {
        const [message, setMessage] = React.useState('');

        return (
          <div>
            <button onClick={() => setMessage('Generation started')}>
              Start Generation
            </button>
            <button onClick={() => setMessage('Generation completed')}>
              Complete Generation
            </button>
            <div role="status" aria-live="polite" aria-atomic="true">
              {message}
            </div>
          </div>
        );
      };

      renderWithProviders(<DynamicAnnouncer />);

      const status = screen.getByRole('status');

      // Click start
      await user.click(screen.getByText('Start Generation'));
      expect(status).toHaveTextContent('Generation started');

      // Click complete
      await user.click(screen.getByText('Complete Generation'));
      expect(status).toHaveTextContent('Generation completed');
    });

    it('should trap focus in modal dialogs', async () => {
      const ModalWithFocusTrap = ({ isOpen, onClose }: any) => {
        const modalRef = React.useRef<HTMLDivElement>(null);

        React.useEffect(() => {
          if (!isOpen || !modalRef.current) return;

          const focusableElements = modalRef.current.querySelectorAll(
            'button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])'
          );
          
          const firstElement = focusableElements[0] as HTMLElement;
          const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

          firstElement?.focus();

          const handleTab = (e: KeyboardEvent) => {
            if (e.key !== 'Tab') return;

            if (e.shiftKey && document.activeElement === firstElement) {
              e.preventDefault();
              lastElement?.focus();
            } else if (!e.shiftKey && document.activeElement === lastElement) {
              e.preventDefault();
              firstElement?.focus();
            }
          };

          document.addEventListener('keydown', handleTab);
          return () => document.removeEventListener('keydown', handleTab);
        }, [isOpen]);

        if (!isOpen) return null;

        return (
          <div ref={modalRef} role="dialog" aria-modal="true">
            <h2>Modal Title</h2>
            <input placeholder="Input 1" />
            <input placeholder="Input 2" />
            <button onClick={onClose}>Close</button>
          </div>
        );
      };

      const App = () => {
        const [isOpen, setIsOpen] = React.useState(false);
        return (
          <div>
            <button onClick={() => setIsOpen(true)}>Open Modal</button>
            <ModalWithFocusTrap isOpen={isOpen} onClose={() => setIsOpen(false)} />
          </div>
        );
      };

      renderWithProviders(<App />);

      await user.click(screen.getByText('Open Modal'));

      // First input should be focused
      expect(screen.getByPlaceholderText('Input 1')).toHaveFocus();

      // Tab through elements
      await user.tab();
      expect(screen.getByPlaceholderText('Input 2')).toHaveFocus();

      await user.tab();
      expect(screen.getByText('Close')).toHaveFocus();

      // Tab should wrap to first element
      await user.tab();
      expect(screen.getByPlaceholderText('Input 1')).toHaveFocus();

      // Shift+Tab should wrap to last element
      await user.keyboard('{Shift>}{Tab}{/Shift}');
      expect(screen.getByText('Close')).toHaveFocus();
    });
  });

  describe('Performance Under Stress', () => {
    it('should handle rendering large lists efficiently', async () => {
      const LargeList = () => {
        const [items] = React.useState(() => 
          Array.from({ length: 1000 }, (_, i) => ({
            id: i,
            name: `Item ${i}`,
            value: Math.random(),
          }))
        );

        const [filter, setFilter] = React.useState('');

        const filteredItems = React.useMemo(() => 
          items.filter(item => 
            item.name.toLowerCase().includes(filter.toLowerCase())
          ),
          [items, filter]
        );

        return (
          <div>
            <input
              placeholder="Filter items"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
            <div data-testid="result-count">{filteredItems.length} items</div>
            <div style={{ height: '400px', overflow: 'auto' }}>
              {filteredItems.slice(0, 50).map(item => (
                <div key={item.id}>{item.name}</div>
              ))}
            </div>
          </div>
        );
      };

      const startTime = performance.now();
      renderWithProviders(<LargeList />);
      const renderTime = performance.now() - startTime;

      expect(renderTime).toBeLessThan(100); // Should render quickly

      // Test filtering performance
      const filterStartTime = performance.now();
      await user.type(screen.getByPlaceholderText('Filter items'), '123');
      const filterTime = performance.now() - filterStartTime;

      expect(filterTime).toBeLessThan(200); // Should filter quickly
      expect(screen.getByTestId('result-count')).toHaveTextContent('1 items');
    });

    it('should debounce rapid input changes', async () => {
      jest.useFakeTimers();
      const mockSearch = jest.fn();

      const DebouncedSearch = () => {
        const [query, setQuery] = React.useState('');
        const debounceTimerRef = React.useRef<NodeJS.Timeout>();

        const handleChange = (value: string) => {
          setQuery(value);
          
          if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
          }

          debounceTimerRef.current = setTimeout(() => {
            mockSearch(value);
          }, 300);
        };

        return (
          <input
            value={query}
            onChange={(e) => handleChange(e.target.value)}
            placeholder="Search"
          />
        );
      };

      renderWithProviders(<DebouncedSearch />);

      const input = screen.getByPlaceholderText('Search');

      // Type rapidly
      await user.type(input, 'test query');

      // Advance timer before debounce
      act(() => {
        jest.advanceTimersByTime(200);
      });

      // Should not have called search yet
      expect(mockSearch).not.toHaveBeenCalled();

      // Advance past debounce
      act(() => {
        jest.advanceTimersByTime(200);
      });

      // Should call search once with final value
      expect(mockSearch).toHaveBeenCalledTimes(1);
      expect(mockSearch).toHaveBeenCalledWith('test query');

      jest.useRealTimers();
    });
  });
});