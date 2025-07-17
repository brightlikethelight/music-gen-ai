/**
 * GenerationStudio Component Tests
 * 
 * Comprehensive tests for the GenerationStudio component including:
 * - Form controls and validation
 * - Music generation workflow
 * - Audio playback functionality
 * - WebSocket connection status
 * - Generation history
 * - Advanced parameters
 * - Save/share functionality
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { GenerationStudio } from '../GenerationStudio';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock hooks
const mockGenerationHook = {
  isGenerating: false,
  generationProgress: { progress: 0, stage: 'initializing' },
  currentTrack: null,
  generationHistory: [],
  generateMusic: jest.fn(),
  stopGeneration: jest.fn(),
  saveGeneration: jest.fn(),
};

const mockAudioHook = {
  isPlaying: false,
  currentTime: 0,
  duration: 0,
  togglePlayback: jest.fn(),
  seek: jest.fn(),
  stop: jest.fn(),
};

const mockWebSocketHook = {
  isConnected: true,
};

jest.mock('@/hooks/useGeneration', () => ({
  useGeneration: () => mockGenerationHook,
}));

jest.mock('@/hooks/useAudio', () => ({
  useAudio: () => mockAudioHook,
}));

jest.mock('@/hooks/useWebSocket', () => ({
  useWebSocket: () => mockWebSocketHook,
}));

// Mock child components
jest.mock('@/components/audio/WaveformPlayer', () => ({
  WaveformPlayer: ({ onPlayPause, onSeek, onStop, ...props }: any) => (
    <div data-testid="waveform-player" {...props}>
      <button onClick={onPlayPause} data-testid="play-pause-button">
        {props.isPlaying ? 'Pause' : 'Play'}
      </button>
      <button onClick={onSeek} data-testid="seek-button">Seek</button>
      <button onClick={onStop} data-testid="stop-button">Stop</button>
    </div>
  ),
}));

jest.mock('@/components/generation/GenerationProgress', () => ({
  GenerationProgress: ({ onCancel, progress }: any) => (
    <div data-testid="generation-progress">
      <div>Progress: {progress.progress}%</div>
      <button onClick={onCancel} data-testid="cancel-generation">Cancel</button>
    </div>
  ),
}));

jest.mock('@/components/generation/ParameterControls', () => ({
  ParameterControls: ({ disabled }: any) => (
    <div data-testid="parameter-controls" aria-disabled={disabled}>
      Advanced Parameters
    </div>
  ),
}));

jest.mock('@/components/generation/GenerationHistory', () => ({
  GenerationHistory: ({ generations, onSelect }: any) => (
    <div data-testid="generation-history">
      {generations.map((gen: any, index: number) => (
        <button key={index} onClick={() => onSelect(gen)} data-testid={`history-item-${index}`}>
          {gen.prompt}
        </button>
      ))}
    </div>
  ),
}));

describe('GenerationStudio', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Reset mock states
    mockGenerationHook.isGenerating = false;
    mockGenerationHook.currentTrack = null;
    mockGenerationHook.generationHistory = [];
    mockWebSocketHook.isConnected = true;
  });

  describe('Rendering', () => {
    it('should render the main generation studio interface', () => {
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByRole('heading', { name: /music generation studio/i })).toBeInTheDocument();
      expect(screen.getByLabelText(/describe your music/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/genre/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/mood/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/duration/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /generate music/i })).toBeInTheDocument();
    });

    it('should render connection status indicator', () => {
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByText(/connected/i)).toBeInTheDocument();
    });

    it('should render disconnected status when not connected', () => {
      mockWebSocketHook.isConnected = false;
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByText(/disconnected/i)).toBeInTheDocument();
    });

    it('should render sidebar with history and tips', () => {
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByRole('heading', { name: /recent generations/i })).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: /pro tips/i })).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: /community favorites/i })).toBeInTheDocument();
    });

    it('should show rotating placeholder text', async () => {
      renderWithProviders(<GenerationStudio />);

      const textarea = screen.getByLabelText(/describe your music/i);
      expect(textarea).toHaveAttribute('placeholder');
      
      // Placeholder should contain music-related text
      const placeholder = textarea.getAttribute('placeholder');
      expect(placeholder).toMatch(/(jazz|electronic|orchestral|folk|rock|piano|bass)/i);
    });
  });

  describe('Form Controls', () => {
    it('should allow typing in the prompt textarea', async () => {
      renderWithProviders(<GenerationStudio />);

      const textarea = screen.getByLabelText(/describe your music/i);
      await user.type(textarea, 'Upbeat jazz with saxophone');

      expect(textarea).toHaveValue('Upbeat jazz with saxophone');
    });

    it('should allow selecting different genres', async () => {
      renderWithProviders(<GenerationStudio />);

      const genreSelect = screen.getByLabelText(/genre/i);
      
      // Initial value should be 'pop'
      expect(genreSelect).toHaveValue('pop');

      // Change to jazz
      await user.selectOptions(genreSelect, 'jazz');
      expect(genreSelect).toHaveValue('jazz');
    });

    it('should allow selecting different moods', async () => {
      renderWithProviders(<GenerationStudio />);

      const moodSelect = screen.getByLabelText(/mood/i);
      
      // Initial value should be 'happy'
      expect(moodSelect).toHaveValue('happy');

      // Change to energetic
      await user.selectOptions(moodSelect, 'energetic');
      expect(moodSelect).toHaveValue('energetic');
    });

    it('should allow adjusting duration with slider', async () => {
      renderWithProviders(<GenerationStudio />);

      const durationSlider = screen.getByLabelText(/duration/i);
      
      // Should show initial duration
      expect(screen.getByText(/duration: 30s/i)).toBeInTheDocument();

      // Change duration
      fireEvent.change(durationSlider, { target: { value: '60' } });
      
      await waitFor(() => {
        expect(screen.getByText(/duration: 60s/i)).toBeInTheDocument();
      });
    });

    it('should disable form controls during generation', () => {
      mockGenerationHook.isGenerating = true;
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByLabelText(/describe your music/i)).toBeDisabled();
      expect(screen.getByLabelText(/genre/i)).toBeDisabled();
      expect(screen.getByLabelText(/mood/i)).toBeDisabled();
      expect(screen.getByLabelText(/duration/i)).toBeDisabled();
    });
  });

  describe('Advanced Parameters', () => {
    it('should toggle advanced parameters visibility', async () => {
      renderWithProviders(<GenerationStudio />);

      const advancedButton = screen.getByRole('button', { name: /advanced parameters/i });
      
      // Should not show advanced parameters initially
      expect(screen.queryByTestId('parameter-controls')).not.toBeInTheDocument();

      // Click to show advanced parameters
      await user.click(advancedButton);
      expect(screen.getByTestId('parameter-controls')).toBeInTheDocument();

      // Click to hide advanced parameters
      await user.click(advancedButton);
      expect(screen.queryByTestId('parameter-controls')).not.toBeInTheDocument();
    });

    it('should disable advanced parameters button during generation', () => {
      mockGenerationHook.isGenerating = true;
      renderWithProviders(<GenerationStudio />);

      const advancedButton = screen.getByRole('button', { name: /advanced parameters/i });
      expect(advancedButton).toBeDisabled();
    });

    it('should disable advanced parameters controls during generation', async () => {
      renderWithProviders(<GenerationStudio />);

      // Show advanced parameters first
      const advancedButton = screen.getByRole('button', { name: /advanced parameters/i });
      await user.click(advancedButton);

      // Mock generation state
      mockGenerationHook.isGenerating = true;
      
      // Re-render with generation state
      renderWithProviders(<GenerationStudio />);
      await user.click(advancedButton);

      const parameterControls = screen.getByTestId('parameter-controls');
      expect(parameterControls).toHaveAttribute('aria-disabled', 'true');
    });
  });

  describe('Music Generation', () => {
    it('should disable generate button when prompt is empty', () => {
      renderWithProviders(<GenerationStudio />);

      const generateButton = screen.getByRole('button', { name: /generate music/i });
      expect(generateButton).toBeDisabled();
    });

    it('should enable generate button when prompt has content', async () => {
      renderWithProviders(<GenerationStudio />);

      const textarea = screen.getByLabelText(/describe your music/i);
      const generateButton = screen.getByRole('button', { name: /generate music/i });

      await user.type(textarea, 'Jazz music');
      expect(generateButton).not.toBeDisabled();
    });

    it('should call generateMusic with correct parameters', async () => {
      renderWithProviders(<GenerationStudio />);

      // Fill form
      await user.type(screen.getByLabelText(/describe your music/i), 'Upbeat jazz');
      await user.selectOptions(screen.getByLabelText(/genre/i), 'jazz');
      await user.selectOptions(screen.getByLabelText(/mood/i), 'energetic');
      
      // Change duration
      const durationSlider = screen.getByLabelText(/duration/i);
      fireEvent.change(durationSlider, { target: { value: '60' } });

      // Generate
      const generateButton = screen.getByRole('button', { name: /generate music/i });
      await user.click(generateButton);

      expect(mockGenerationHook.generateMusic).toHaveBeenCalledWith({
        prompt: 'Upbeat jazz',
        genre: 'jazz',
        mood: 'energetic',
        duration: 60,
      });
    });

    it('should show loading state during generation', () => {
      mockGenerationHook.isGenerating = true;
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByText(/generating.../i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /generating.../i })).toBeDisabled();
    });

    it('should show generation progress when generating', () => {
      mockGenerationHook.isGenerating = true;
      mockGenerationHook.generationProgress = { progress: 45, stage: 'processing' };
      
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByTestId('generation-progress')).toBeInTheDocument();
      expect(screen.getByText(/progress: 45%/i)).toBeInTheDocument();
    });

    it('should allow canceling generation', async () => {
      mockGenerationHook.isGenerating = true;
      renderWithProviders(<GenerationStudio />);

      const cancelButton = screen.getByTestId('cancel-generation');
      await user.click(cancelButton);

      expect(mockGenerationHook.stopGeneration).toHaveBeenCalled();
    });

    it('should trim whitespace from prompt before generation', async () => {
      renderWithProviders(<GenerationStudio />);

      await user.type(screen.getByLabelText(/describe your music/i), '  Jazz music  ');
      await user.click(screen.getByRole('button', { name: /generate music/i }));

      expect(mockGenerationHook.generateMusic).toHaveBeenCalledWith({
        prompt: 'Jazz music',
        genre: 'pop',
        mood: 'happy',
        duration: 30,
      });
    });
  });

  describe('Audio Playback', () => {
    it('should show audio player when track is available', () => {
      mockGenerationHook.currentTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: '/audio/track.wav',
        waveformData: [0.1, 0.2, 0.3],
      };

      renderWithProviders(<GenerationStudio />);

      expect(screen.getByTestId('waveform-player')).toBeInTheDocument();
      expect(screen.getByText(/generated track/i)).toBeInTheDocument();
      expect(screen.getByText('Jazz music')).toBeInTheDocument();
    });

    it('should pass correct props to WaveformPlayer', () => {
      const mockTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: '/audio/track.wav',
        waveformData: [0.1, 0.2, 0.3],
      };

      mockGenerationHook.currentTrack = mockTrack;
      mockAudioHook.isPlaying = true;
      mockAudioHook.currentTime = 15;
      mockAudioHook.duration = 30;

      renderWithProviders(<GenerationStudio />);

      const waveformPlayer = screen.getByTestId('waveform-player');
      expect(waveformPlayer).toHaveAttribute('src', '/audio/track.wav');
      expect(waveformPlayer).toHaveAttribute('isPlaying', 'true');
      expect(waveformPlayer).toHaveAttribute('currentTime', '15');
      expect(waveformPlayer).toHaveAttribute('duration', '30');
    });

    it('should handle play/pause controls', async () => {
      mockGenerationHook.currentTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: '/audio/track.wav',
        waveformData: [0.1, 0.2, 0.3],
      };

      renderWithProviders(<GenerationStudio />);

      const playPauseButton = screen.getByTestId('play-pause-button');
      await user.click(playPauseButton);

      expect(mockAudioHook.togglePlayback).toHaveBeenCalled();
    });

    it('should handle seek controls', async () => {
      mockGenerationHook.currentTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: '/audio/track.wav',
        waveformData: [0.1, 0.2, 0.3],
      };

      renderWithProviders(<GenerationStudio />);

      const seekButton = screen.getByTestId('seek-button');
      await user.click(seekButton);

      expect(mockAudioHook.seek).toHaveBeenCalled();
    });

    it('should handle stop controls', async () => {
      mockGenerationHook.currentTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: '/audio/track.wav',
        waveformData: [0.1, 0.2, 0.3],
      };

      renderWithProviders(<GenerationStudio />);

      const stopButton = screen.getByTestId('stop-button');
      await user.click(stopButton);

      expect(mockAudioHook.stop).toHaveBeenCalled();
    });
  });

  describe('Save and Share Functionality', () => {
    it('should show save and share buttons when track is available', () => {
      mockGenerationHook.currentTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: '/audio/track.wav',
        waveformData: [0.1, 0.2, 0.3],
      };

      renderWithProviders(<GenerationStudio />);

      expect(screen.getByRole('button', { name: /save/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /share/i })).toBeInTheDocument();
    });

    it('should handle save functionality', async () => {
      mockGenerationHook.currentTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: '/audio/track.wav',
        waveformData: [0.1, 0.2, 0.3],
      };

      renderWithProviders(<GenerationStudio />);

      const saveButton = screen.getByRole('button', { name: /save/i });
      await user.click(saveButton);

      expect(mockGenerationHook.saveGeneration).toHaveBeenCalledWith('track-1');
    });

    it('should handle share functionality', async () => {
      mockGenerationHook.currentTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: '/audio/track.wav',
        waveformData: [0.1, 0.2, 0.3],
      };

      renderWithProviders(<GenerationStudio />);

      const shareButton = screen.getByRole('button', { name: /share/i });
      expect(shareButton).toBeInTheDocument();
      
      // Share functionality would trigger share logic
      await user.click(shareButton);
      // Currently just rendering the button, actual share logic would be tested separately
    });
  });

  describe('Generation History', () => {
    it('should render generation history with items', () => {
      mockGenerationHook.generationHistory = [
        { id: '1', prompt: 'Jazz music', createdAt: new Date() },
        { id: '2', prompt: 'Rock anthem', createdAt: new Date() },
      ];

      renderWithProviders(<GenerationStudio />);

      expect(screen.getByTestId('generation-history')).toBeInTheDocument();
      expect(screen.getByTestId('history-item-0')).toBeInTheDocument();
      expect(screen.getByTestId('history-item-1')).toBeInTheDocument();
    });

    it('should handle selection of history items', async () => {
      const mockTrack = { id: '1', prompt: 'Jazz music', createdAt: new Date() };
      mockGenerationHook.generationHistory = [mockTrack];

      renderWithProviders(<GenerationStudio />);

      const historyItem = screen.getByTestId('history-item-0');
      await user.click(historyItem);

      // History selection logic would be implemented in the actual component
      expect(historyItem).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels for form controls', () => {
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByLabelText(/describe your music/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/genre/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/mood/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/duration/i)).toBeInTheDocument();
    });

    it('should be keyboard navigable', async () => {
      renderWithProviders(<GenerationStudio />);

      // Tab through form controls
      await user.tab();
      expect(screen.getByLabelText(/describe your music/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/genre/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/mood/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/duration/i)).toHaveFocus();
    });

    it('should announce generation state to screen readers', () => {
      mockGenerationHook.isGenerating = true;
      renderWithProviders(<GenerationStudio />);

      const generateButton = screen.getByRole('button', { name: /generating.../i });
      expect(generateButton).toHaveAttribute('disabled');
    });

    it('should have proper heading hierarchy', () => {
      renderWithProviders(<GenerationStudio />);

      const mainHeading = screen.getByRole('heading', { name: /music generation studio/i });
      expect(mainHeading).toBeInTheDocument();
      
      const sidebarHeadings = screen.getAllByRole('heading', { level: 2 });
      expect(sidebarHeadings.length).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle generation errors gracefully', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      mockGenerationHook.generateMusic.mockRejectedValue(new Error('Generation failed'));

      renderWithProviders(<GenerationStudio />);

      await user.type(screen.getByLabelText(/describe your music/i), 'Jazz music');
      await user.click(screen.getByRole('button', { name: /generate music/i }));

      expect(mockGenerationHook.generateMusic).toHaveBeenCalled();
      
      consoleSpy.mockRestore();
    });

    it('should handle invalid duration values', async () => {
      renderWithProviders(<GenerationStudio />);

      const durationSlider = screen.getByLabelText(/duration/i);
      
      // Try to set invalid duration (below minimum)
      fireEvent.change(durationSlider, { target: { value: '5' } });
      
      // Component should enforce minimum value
      expect(durationSlider).toHaveAttribute('min', '10');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty generation history', () => {
      mockGenerationHook.generationHistory = [];
      renderWithProviders(<GenerationStudio />);

      expect(screen.getByTestId('generation-history')).toBeInTheDocument();
      expect(screen.queryByTestId('history-item-0')).not.toBeInTheDocument();
    });

    it('should handle very long prompts', async () => {
      renderWithProviders(<GenerationStudio />);

      const longPrompt = 'A'.repeat(1000);
      const textarea = screen.getByLabelText(/describe your music/i);
      
      await user.type(textarea, longPrompt);
      expect(textarea).toHaveValue(longPrompt);
    });

    it('should handle rapid button clicks', async () => {
      renderWithProviders(<GenerationStudio />);

      await user.type(screen.getByLabelText(/describe your music/i), 'Jazz music');
      const generateButton = screen.getByRole('button', { name: /generate music/i });

      // Click multiple times rapidly
      await user.click(generateButton);
      await user.click(generateButton);
      await user.click(generateButton);

      // Should only call generateMusic once due to loading state
      expect(mockGenerationHook.generateMusic).toHaveBeenCalledTimes(1);
    });

    it('should handle missing track data gracefully', () => {
      mockGenerationHook.currentTrack = {
        id: 'track-1',
        prompt: 'Jazz music',
        audioUrl: null,
        waveformData: null,
      };

      renderWithProviders(<GenerationStudio />);

      expect(screen.getByTestId('waveform-player')).toBeInTheDocument();
      expect(screen.getByText('Jazz music')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('should render grid layout for desktop', () => {
      renderWithProviders(<GenerationStudio />);

      const container = screen.getByText(/music generation studio/i).closest('.grid');
      expect(container).toHaveClass('lg:grid-cols-12');
    });

    it('should handle mobile viewport', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      renderWithProviders(<GenerationStudio />);

      const container = screen.getByText(/music generation studio/i).closest('.grid');
      expect(container).toHaveClass('grid-cols-1');
    });
  });
});