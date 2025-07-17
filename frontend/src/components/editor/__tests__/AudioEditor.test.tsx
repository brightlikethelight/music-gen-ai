/**
 * AudioEditor Component Tests
 * 
 * Comprehensive tests for the AudioEditor component including:
 * - Transport controls (play, pause, stop, seek)
 * - File operations (open, save, import, export)
 * - Tool selection and editing tools
 * - Panel management (tracks, effects, mixer, tools)
 * - Timeline interactions
 * - Tempo and time signature controls
 * - Undo/redo functionality
 * - Zoom controls
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { AudioEditor } from '../AudioEditor';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock useAudioEditor hook
const mockAudioEditorHook = {
  tracks: [],
  selectedTrack: null,
  addTrack: jest.fn(),
  removeTrack: jest.fn(),
  updateTrack: jest.fn(),
  cutSelection: jest.fn(),
  copySelection: jest.fn(),
  pasteSelection: jest.fn(),
  undoAction: jest.fn(),
  redoAction: jest.fn(),
  canUndo: false,
  canRedo: false,
};

jest.mock('@/hooks/useAudioEditor', () => ({
  useAudioEditor: () => mockAudioEditorHook,
}));

// Mock formatTime utility
jest.mock('@/utils/time', () => ({
  formatTime: (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  },
}));

// Mock child components
jest.mock('@/components/editor/Timeline', () => ({
  Timeline: ({ onTimeChange, onTrackUpdate, ...props }: any) => (
    <div data-testid="timeline" {...props}>
      <button onClick={() => onTimeChange(30)} data-testid="timeline-click">
        Click Timeline
      </button>
      <button onClick={() => onTrackUpdate('track-1', {})} data-testid="track-update">
        Update Track
      </button>
    </div>
  ),
}));

jest.mock('@/components/editor/TrackPanel', () => ({
  TrackPanel: ({ onAddTrack, onRemoveTrack, onUpdateTrack, ...props }: any) => (
    <div data-testid="track-panel" {...props}>
      <button onClick={() => onAddTrack({ type: 'audio' })} data-testid="add-track">
        Add Track
      </button>
      <button onClick={() => onRemoveTrack('track-1')} data-testid="remove-track">
        Remove Track
      </button>
      <button onClick={() => onUpdateTrack('track-1', {})} data-testid="update-track">
        Update Track
      </button>
    </div>
  ),
}));

jest.mock('@/components/editor/EffectsPanel', () => ({
  EffectsPanel: ({ onUpdateTrack, ...props }: any) => (
    <div data-testid="effects-panel" {...props}>
      <button onClick={() => onUpdateTrack('track-1', {})} data-testid="update-effect">
        Update Effect
      </button>
    </div>
  ),
}));

jest.mock('@/components/editor/MixerPanel', () => ({
  MixerPanel: ({ onUpdateTrack, ...props }: any) => (
    <div data-testid="mixer-panel" {...props}>
      <button onClick={() => onUpdateTrack('track-1', {})} data-testid="update-mixer">
        Update Mixer
      </button>
    </div>
  ),
}));

jest.mock('@/components/editor/ToolsPanel', () => ({
  ToolsPanel: ({ onToolChange, onCut, onCopy, onPaste, ...props }: any) => (
    <div data-testid="tools-panel" {...props}>
      <button onClick={() => onToolChange('cut')} data-testid="change-tool">
        Change Tool
      </button>
      <button onClick={onCut} data-testid="cut-selection">
        Cut
      </button>
      <button onClick={onCopy} data-testid="copy-selection">
        Copy
      </button>
      <button onClick={onPaste} data-testid="paste-selection">
        Paste
      </button>
    </div>
  ),
}));

jest.mock('@/components/editor/ExportDialog', () => ({
  ExportDialog: ({ isOpen, onClose, ...props }: any) => (
    isOpen ? (
      <div data-testid="export-dialog" {...props}>
        <button onClick={onClose} data-testid="close-export">
          Close Export
        </button>
      </div>
    ) : null
  ),
}));

describe('AudioEditor', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Reset mock states
    mockAudioEditorHook.tracks = [];
    mockAudioEditorHook.selectedTrack = null;
    mockAudioEditorHook.canUndo = false;
    mockAudioEditorHook.canRedo = false;
  });

  describe('Rendering', () => {
    it('should render the audio editor interface', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('heading', { name: /audio editor/i })).toBeInTheDocument();
      expect(screen.getByTestId('timeline')).toBeInTheDocument();
    });

    it('should render top menu bar with file operations', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('button', { name: /open/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /save/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /import/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /export/i })).toBeInTheDocument();
    });

    it('should render transport controls', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
      expect(screen.getByTitle(/rewind/i)).toBeInTheDocument();
      expect(screen.getByTitle(/forward/i)).toBeInTheDocument();
    });

    it('should render tempo and time signature controls', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByDisplayValue('120')).toBeInTheDocument(); // Tempo
      expect(screen.getByDisplayValue('4/4')).toBeInTheDocument(); // Time signature
    });

    it('should render tool selection buttons', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByTitle('Select')).toBeInTheDocument();
      expect(screen.getByTitle('Cut')).toBeInTheDocument();
      expect(screen.getByTitle('Fade')).toBeInTheDocument();
      expect(screen.getByTitle('Pitch')).toBeInTheDocument();
      expect(screen.getByTitle('Time Stretch')).toBeInTheDocument();
    });

    it('should render panel tabs', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('button', { name: /tracks/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /effects/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /mixer/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /tools/i })).toBeInTheDocument();
    });

    it('should render time display', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByText('0:00 / 3:00')).toBeInTheDocument();
    });

    it('should render zoom controls', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByText('100%')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /zoom in/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /zoom out/i })).toBeInTheDocument();
    });
  });

  describe('Transport Controls', () => {
    it('should toggle play/pause state', async () => {
      renderWithProviders(<AudioEditor />);

      const playButton = screen.getByRole('button', { name: /play/i });
      await user.click(playButton);

      // After clicking play, button should show pause
      expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument();
    });

    it('should handle stop button', async () => {
      renderWithProviders(<AudioEditor />);

      // Start playing first
      const playButton = screen.getByRole('button', { name: /play/i });
      await user.click(playButton);
      
      // Then stop
      const stopButton = screen.getByRole('button', { name: /stop/i });
      await user.click(stopButton);

      // Should return to play state and reset time
      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
      expect(screen.getByText('0:00 / 3:00')).toBeInTheDocument();
    });

    it('should handle rewind button', async () => {
      renderWithProviders(<AudioEditor />);

      // First move forward in time (simulate)
      const timelineClick = screen.getByTestId('timeline-click');
      await user.click(timelineClick); // Sets time to 30 seconds

      await waitFor(() => {
        expect(screen.getByText('0:30 / 3:00')).toBeInTheDocument();
      });

      // Then rewind
      const rewindButton = screen.getByTitle(/rewind/i);
      await user.click(rewindButton);

      await waitFor(() => {
        expect(screen.getByText('0:20 / 3:00')).toBeInTheDocument();
      });
    });

    it('should handle forward button', async () => {
      renderWithProviders(<AudioEditor />);

      const forwardButton = screen.getByTitle(/forward/i);
      await user.click(forwardButton);

      await waitFor(() => {
        expect(screen.getByText('0:10 / 3:00')).toBeInTheDocument();
      });
    });

    it('should not rewind below 0 seconds', async () => {
      renderWithProviders(<AudioEditor />);

      const rewindButton = screen.getByTitle(/rewind/i);
      await user.click(rewindButton);

      // Should stay at 0:00
      expect(screen.getByText('0:00 / 3:00')).toBeInTheDocument();
    });

    it('should not forward beyond duration', async () => {
      renderWithProviders(<AudioEditor />);

      // Click forward many times to exceed duration
      const forwardButton = screen.getByTitle(/forward/i);
      for (let i = 0; i < 20; i++) {
        await user.click(forwardButton);
      }

      // Should stay at 3:00 (duration)
      await waitFor(() => {
        expect(screen.getByText('3:00 / 3:00')).toBeInTheDocument();
      });
    });
  });

  describe('File Operations', () => {
    it('should handle open project', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      renderWithProviders(<AudioEditor />);

      const openButton = screen.getByRole('button', { name: /open/i });
      await user.click(openButton);

      expect(consoleSpy).toHaveBeenCalledWith('Opening project...');
      consoleSpy.mockRestore();
    });

    it('should handle save project', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      renderWithProviders(<AudioEditor />);

      const saveButton = screen.getByRole('button', { name: /save/i });
      await user.click(saveButton);

      expect(consoleSpy).toHaveBeenCalledWith('Saving project...');
      consoleSpy.mockRestore();
    });

    it('should handle import audio', async () => {
      const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
      
      renderWithProviders(<AudioEditor />);

      const importButton = screen.getByRole('button', { name: /import/i });
      await user.click(importButton);

      expect(consoleSpy).toHaveBeenCalledWith('Importing audio...');
      consoleSpy.mockRestore();
    });

    it('should open export dialog', async () => {
      renderWithProviders(<AudioEditor />);

      const exportButton = screen.getByRole('button', { name: /export/i });
      await user.click(exportButton);

      expect(screen.getByTestId('export-dialog')).toBeInTheDocument();
    });

    it('should close export dialog', async () => {
      renderWithProviders(<AudioEditor />);

      // Open export dialog
      const exportButton = screen.getByRole('button', { name: /export/i });
      await user.click(exportButton);

      // Close export dialog
      const closeButton = screen.getByTestId('close-export');
      await user.click(closeButton);

      expect(screen.queryByTestId('export-dialog')).not.toBeInTheDocument();
    });
  });

  describe('Tempo and Time Signature Controls', () => {
    it('should update tempo', async () => {
      renderWithProviders(<AudioEditor />);

      const tempoInput = screen.getByDisplayValue('120');
      await user.clear(tempoInput);
      await user.type(tempoInput, '140');

      expect(tempoInput).toHaveValue(140);
    });

    it('should enforce tempo limits', async () => {
      renderWithProviders(<AudioEditor />);

      const tempoInput = screen.getByDisplayValue('120');
      expect(tempoInput).toHaveAttribute('min', '60');
      expect(tempoInput).toHaveAttribute('max', '200');
    });

    it('should update time signature', async () => {
      renderWithProviders(<AudioEditor />);

      const timeSignatureSelect = screen.getByDisplayValue('4/4');
      await user.selectOptions(timeSignatureSelect, '3/4');

      expect(timeSignatureSelect).toHaveValue('3/4');
    });

    it('should have all time signature options', () => {
      renderWithProviders(<AudioEditor />);

      const timeSignatureSelect = screen.getByDisplayValue('4/4');
      const options = Array.from(timeSignatureSelect.querySelectorAll('option')).map(
        option => option.textContent
      );

      expect(options).toEqual(['4/4', '3/4', '6/8', '2/4']);
    });
  });

  describe('Undo/Redo Functionality', () => {
    it('should disable undo/redo buttons when not available', () => {
      mockAudioEditorHook.canUndo = false;
      mockAudioEditorHook.canRedo = false;
      
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('button', { name: '↶' })).toBeDisabled();
      expect(screen.getByRole('button', { name: '↷' })).toBeDisabled();
    });

    it('should enable undo/redo buttons when available', () => {
      mockAudioEditorHook.canUndo = true;
      mockAudioEditorHook.canRedo = true;
      
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('button', { name: '↶' })).not.toBeDisabled();
      expect(screen.getByRole('button', { name: '↷' })).not.toBeDisabled();
    });

    it('should handle undo action', async () => {
      mockAudioEditorHook.canUndo = true;
      
      renderWithProviders(<AudioEditor />);

      const undoButton = screen.getByRole('button', { name: '↶' });
      await user.click(undoButton);

      expect(mockAudioEditorHook.undoAction).toHaveBeenCalled();
    });

    it('should handle redo action', async () => {
      mockAudioEditorHook.canRedo = true;
      
      renderWithProviders(<AudioEditor />);

      const redoButton = screen.getByRole('button', { name: '↷' });
      await user.click(redoButton);

      expect(mockAudioEditorHook.redoAction).toHaveBeenCalled();
    });
  });

  describe('Tool Selection', () => {
    it('should select tools', async () => {
      renderWithProviders(<AudioEditor />);

      const cutTool = screen.getByTitle('Cut');
      await user.click(cutTool);

      expect(cutTool).toHaveClass('bg-blue-600');
    });

    it('should highlight selected tool', () => {
      renderWithProviders(<AudioEditor />);

      const selectTool = screen.getByTitle('Select');
      expect(selectTool).toHaveClass('bg-blue-600'); // Default selected
    });

    it('should deselect previous tool when selecting new one', async () => {
      renderWithProviders(<AudioEditor />);

      const selectTool = screen.getByTitle('Select');
      const cutTool = screen.getByTitle('Cut');

      await user.click(cutTool);

      expect(selectTool).not.toHaveClass('bg-blue-600');
      expect(cutTool).toHaveClass('bg-blue-600');
    });
  });

  describe('Panel Management', () => {
    it('should switch between panels', async () => {
      renderWithProviders(<AudioEditor />);

      // Initially tracks panel should be active
      expect(screen.getByTestId('track-panel')).toBeInTheDocument();

      // Switch to effects panel
      const effectsTab = screen.getByRole('button', { name: /effects/i });
      await user.click(effectsTab);

      expect(screen.getByTestId('effects-panel')).toBeInTheDocument();
      expect(screen.queryByTestId('track-panel')).not.toBeInTheDocument();
    });

    it('should highlight active panel tab', async () => {
      renderWithProviders(<AudioEditor />);

      const tracksTab = screen.getByRole('button', { name: /tracks/i });
      const effectsTab = screen.getByRole('button', { name: /effects/i });

      expect(tracksTab).toHaveClass('bg-gray-700');

      await user.click(effectsTab);

      expect(effectsTab).toHaveClass('bg-gray-700');
      expect(tracksTab).not.toHaveClass('bg-gray-700');
    });

    it('should render all panel types', async () => {
      renderWithProviders(<AudioEditor />);

      // Test tracks panel
      expect(screen.getByTestId('track-panel')).toBeInTheDocument();

      // Test effects panel
      await user.click(screen.getByRole('button', { name: /effects/i }));
      expect(screen.getByTestId('effects-panel')).toBeInTheDocument();

      // Test mixer panel
      await user.click(screen.getByRole('button', { name: /mixer/i }));
      expect(screen.getByTestId('mixer-panel')).toBeInTheDocument();

      // Test tools panel
      await user.click(screen.getByRole('button', { name: /tools/i }));
      expect(screen.getByTestId('tools-panel')).toBeInTheDocument();
    });
  });

  describe('Zoom Controls', () => {
    it('should zoom in', async () => {
      renderWithProviders(<AudioEditor />);

      const zoomInButton = screen.getByRole('button', { name: /zoom in/i });
      await user.click(zoomInButton);

      expect(screen.getByText('150%')).toBeInTheDocument();
    });

    it('should zoom out', async () => {
      renderWithProviders(<AudioEditor />);

      const zoomOutButton = screen.getByRole('button', { name: /zoom out/i });
      await user.click(zoomOutButton);

      expect(screen.getByText('67%')).toBeInTheDocument();
    });

    it('should limit maximum zoom', async () => {
      renderWithProviders(<AudioEditor />);

      const zoomInButton = screen.getByRole('button', { name: /zoom in/i });
      
      // Click zoom in many times
      for (let i = 0; i < 10; i++) {
        await user.click(zoomInButton);
      }

      expect(screen.getByText('800%')).toBeInTheDocument(); // Max 8x zoom
    });

    it('should limit minimum zoom', async () => {
      renderWithProviders(<AudioEditor />);

      const zoomOutButton = screen.getByRole('button', { name: /zoom out/i });
      
      // Click zoom out many times
      for (let i = 0; i < 10; i++) {
        await user.click(zoomOutButton);
      }

      expect(screen.getByText('25%')).toBeInTheDocument(); // Min 0.25x zoom
    });
  });

  describe('Timeline Interactions', () => {
    it('should handle timeline clicks', async () => {
      renderWithProviders(<AudioEditor />);

      const timelineClick = screen.getByTestId('timeline-click');
      await user.click(timelineClick);

      await waitFor(() => {
        expect(screen.getByText('0:30 / 3:00')).toBeInTheDocument();
      });
    });

    it('should pass correct props to Timeline', () => {
      mockAudioEditorHook.tracks = [
        { id: 'track-1', name: 'Track 1', type: 'audio' },
        { id: 'track-2', name: 'Track 2', type: 'midi' },
      ];

      renderWithProviders(<AudioEditor />);

      const timeline = screen.getByTestId('timeline');
      expect(timeline).toHaveAttribute('duration', '180');
      expect(timeline).toHaveAttribute('currentTime', '0');
      expect(timeline).toHaveAttribute('zoomLevel', '1');
      expect(timeline).toHaveAttribute('selectedTool', 'select');
      expect(timeline).toHaveAttribute('isPlaying', 'false');
    });

    it('should handle track updates from timeline', async () => {
      renderWithProviders(<AudioEditor />);

      const trackUpdate = screen.getByTestId('track-update');
      await user.click(trackUpdate);

      expect(mockAudioEditorHook.updateTrack).toHaveBeenCalledWith('track-1', {});
    });
  });

  describe('Track Management', () => {
    it('should handle track operations from TrackPanel', async () => {
      renderWithProviders(<AudioEditor />);

      // Test add track
      const addTrackButton = screen.getByTestId('add-track');
      await user.click(addTrackButton);
      expect(mockAudioEditorHook.addTrack).toHaveBeenCalledWith({ type: 'audio' });

      // Test remove track
      const removeTrackButton = screen.getByTestId('remove-track');
      await user.click(removeTrackButton);
      expect(mockAudioEditorHook.removeTrack).toHaveBeenCalledWith('track-1');

      // Test update track
      const updateTrackButton = screen.getByTestId('update-track');
      await user.click(updateTrackButton);
      expect(mockAudioEditorHook.updateTrack).toHaveBeenCalledWith('track-1', {});
    });

    it('should handle operations from ToolsPanel', async () => {
      renderWithProviders(<AudioEditor />);

      // Switch to tools panel
      await user.click(screen.getByRole('button', { name: /tools/i }));

      // Test cut selection
      const cutButton = screen.getByTestId('cut-selection');
      await user.click(cutButton);
      expect(mockAudioEditorHook.cutSelection).toHaveBeenCalled();

      // Test copy selection
      const copyButton = screen.getByTestId('copy-selection');
      await user.click(copyButton);
      expect(mockAudioEditorHook.copySelection).toHaveBeenCalled();

      // Test paste selection
      const pasteButton = screen.getByTestId('paste-selection');
      await user.click(pasteButton);
      expect(mockAudioEditorHook.pasteSelection).toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('heading', { name: /audio editor/i })).toBeInTheDocument();
      expect(screen.getByLabelText(/tempo/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/time signature/i)).toBeInTheDocument();
    });

    it('should be keyboard navigable', async () => {
      renderWithProviders(<AudioEditor />);

      // Tab through interactive elements
      await user.tab();
      expect(screen.getByRole('button', { name: /open/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /save/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /import/i })).toHaveFocus();
    });

    it('should have proper button states', () => {
      mockAudioEditorHook.canUndo = false;
      mockAudioEditorHook.canRedo = false;
      
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('button', { name: '↶' })).toBeDisabled();
      expect(screen.getByRole('button', { name: '↷' })).toBeDisabled();
    });

    it('should have proper tool button attributes', () => {
      renderWithProviders(<AudioEditor />);

      const selectTool = screen.getByTitle('Select');
      expect(selectTool).toHaveAttribute('title', 'Select');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty tracks array', () => {
      mockAudioEditorHook.tracks = [];
      
      renderWithProviders(<AudioEditor />);

      expect(screen.getByTestId('timeline')).toBeInTheDocument();
      expect(screen.getByTestId('track-panel')).toBeInTheDocument();
    });

    it('should handle invalid tempo values', async () => {
      renderWithProviders(<AudioEditor />);

      const tempoInput = screen.getByDisplayValue('120');
      
      // Try to set tempo below minimum
      await user.clear(tempoInput);
      await user.type(tempoInput, '50');

      // Input should enforce min/max constraints
      expect(tempoInput).toHaveAttribute('min', '60');
      expect(tempoInput).toHaveAttribute('max', '200');
    });

    it('should handle rapid button clicks', async () => {
      renderWithProviders(<AudioEditor />);

      const playButton = screen.getByRole('button', { name: /play/i });
      
      // Click multiple times rapidly
      await user.click(playButton);
      await user.click(playButton);
      await user.click(playButton);

      // Should maintain consistent state
      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('should handle missing hook data gracefully', () => {
      mockAudioEditorHook.tracks = [];
      mockAudioEditorHook.selectedTrack = null;
      
      renderWithProviders(<AudioEditor />);

      expect(screen.getByRole('heading', { name: /audio editor/i })).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('should hide panel text on small screens', () => {
      renderWithProviders(<AudioEditor />);

      const panelButtons = screen.getAllByRole('button').filter(button => 
        button.textContent?.includes('Tracks') || 
        button.textContent?.includes('Effects') ||
        button.textContent?.includes('Mixer') ||
        button.textContent?.includes('Tools')
      );

      panelButtons.forEach(button => {
        const textSpan = button.querySelector('span');
        if (textSpan) {
          expect(textSpan).toHaveClass('hidden', 'sm:inline');
        }
      });
    });

    it('should maintain fixed sidebar width', () => {
      renderWithProviders(<AudioEditor />);

      const sidebar = screen.getByTestId('track-panel').closest('.w-80');
      expect(sidebar).toHaveClass('w-80');
    });
  });
});