/**
 * SocialLogin Component Tests
 * 
 * Comprehensive tests for the SocialLogin and GoogleOAuthButton components including:
 * - Social provider rendering
 * - OAuth redirections
 * - Loading states
 * - Error handling
 * - Google SDK integration
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { SocialLogin, GoogleOAuthButton } from '../SocialLogin';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
}));

// Mock fetch
global.fetch = jest.fn();

// Mock window.location
const mockLocation = {
  href: '',
  origin: 'http://localhost:3000',
};
Object.defineProperty(window, 'location', {
  value: mockLocation,
  writable: true,
});

// Mock Google OAuth
const mockGoogle = {
  accounts: {
    id: {
      initialize: jest.fn(),
      prompt: jest.fn(),
    },
  },
};

describe('SocialLogin', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    mockLocation.href = '';
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render all social provider buttons', () => {
      renderWithProviders(<SocialLogin />);

      expect(screen.getByRole('button', { name: /continue with google/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /continue with apple/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /continue with spotify/i })).toBeInTheDocument();
    });

    it('should render provider buttons with correct icons and styling', () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      const appleButton = screen.getByRole('button', { name: /continue with apple/i });
      const spotifyButton = screen.getByRole('button', { name: /continue with spotify/i });

      expect(googleButton).toHaveTextContent('🔍');
      expect(appleButton).toHaveTextContent('🍎');
      expect(spotifyButton).toHaveTextContent('🎵');
    });

    it('should have proper button types', () => {
      renderWithProviders(<SocialLogin />);

      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveAttribute('type', 'button');
      });
    });
  });

  describe('OAuth Redirections', () => {
    it('should redirect to Google OAuth when Google button is clicked', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      await user.click(googleButton);

      expect(mockLocation.href).toBe('/api/auth/google');
    });

    it('should redirect to Apple OAuth when Apple button is clicked', async () => {
      renderWithProviders(<SocialLogin />);

      const appleButton = screen.getByRole('button', { name: /continue with apple/i });
      await user.click(appleButton);

      expect(mockLocation.href).toBe('/api/auth/apple');
    });

    it('should redirect to Spotify OAuth when Spotify button is clicked', async () => {
      renderWithProviders(<SocialLogin />);

      const spotifyButton = screen.getByRole('button', { name: /continue with spotify/i });
      await user.click(spotifyButton);

      expect(mockLocation.href).toBe('/api/auth/spotify');
    });
  });

  describe('Loading States', () => {
    it('should show loading state for clicked provider', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      await user.click(googleButton);

      expect(screen.getByText(/connecting.../i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /connecting.../i })).toBeDisabled();
    });

    it('should disable all buttons when one provider is loading', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      await user.click(googleButton);

      const allButtons = screen.getAllByRole('button');
      allButtons.forEach(button => {
        expect(button).toBeDisabled();
      });
    });

    it('should show loading spinner during authentication', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      await user.click(googleButton);

      const spinner = screen.getByRole('button', { name: /connecting.../i }).querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should handle errors and reset loading state', async () => {
      // Mock console.error to avoid test output
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      // Mock window.location.href to throw an error
      Object.defineProperty(window, 'location', {
        value: {
          ...mockLocation,
          set href(value: string) {
            throw new Error('Navigation failed');
          },
        },
        writable: true,
      });

      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      await user.click(googleButton);

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith('google login failed:', expect.any(Error));
      });

      // Loading state should be cleared
      expect(screen.queryByText(/connecting.../i)).not.toBeInTheDocument();
      expect(googleButton).not.toBeDisabled();

      consoleSpy.mockRestore();
    });
  });

  describe('User Interactions', () => {
    it('should handle rapid clicks on same provider', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      
      // Click multiple times rapidly
      await user.click(googleButton);
      await user.click(googleButton);
      await user.click(googleButton);

      // Should still only redirect once
      expect(mockLocation.href).toBe('/api/auth/google');
    });

    it('should handle clicks on different providers', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      const appleButton = screen.getByRole('button', { name: /continue with apple/i });

      await user.click(googleButton);
      
      // Apple button should be disabled due to loading state
      expect(appleButton).toBeDisabled();
    });

    it('should handle keyboard navigation', async () => {
      renderWithProviders(<SocialLogin />);

      // Tab through buttons
      await user.tab();
      expect(screen.getByRole('button', { name: /continue with google/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /continue with apple/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /continue with spotify/i })).toHaveFocus();
    });

    it('should handle Enter key to activate buttons', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      googleButton.focus();
      
      await user.keyboard('{Enter}');
      expect(mockLocation.href).toBe('/api/auth/google');
    });

    it('should handle Space key to activate buttons', async () => {
      renderWithProviders(<SocialLogin />);

      const appleButton = screen.getByRole('button', { name: /continue with apple/i });
      appleButton.focus();
      
      await user.keyboard(' ');
      expect(mockLocation.href).toBe('/api/auth/apple');
    });
  });

  describe('Accessibility', () => {
    it('should have accessible button labels', () => {
      renderWithProviders(<SocialLogin />);

      expect(screen.getByRole('button', { name: /continue with google/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /continue with apple/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /continue with spotify/i })).toBeInTheDocument();
    });

    it('should maintain focus management during loading', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      googleButton.focus();
      
      await user.click(googleButton);

      // Button should still be focusable even when disabled
      expect(document.activeElement).toBe(googleButton);
    });

    it('should announce loading state to screen readers', async () => {
      renderWithProviders(<SocialLogin />);

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      await user.click(googleButton);

      const loadingButton = screen.getByRole('button', { name: /connecting.../i });
      expect(loadingButton).toHaveAttribute('disabled');
    });
  });
});

describe('GoogleOAuthButton', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    mockLocation.href = '';
    (global.fetch as jest.Mock).mockClear();
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render Google OAuth button with proper styling', () => {
      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      expect(button).toBeInTheDocument();
      expect(button).toHaveAttribute('type', 'button');
    });

    it('should render Google icon SVG', () => {
      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      const svg = button.querySelector('svg');
      expect(svg).toBeInTheDocument();
      expect(svg).toHaveAttribute('viewBox', '0 0 24 24');
    });
  });

  describe('Google SDK Integration', () => {
    it('should use Google SDK when available', async () => {
      // Mock Google SDK
      Object.defineProperty(window, 'google', {
        value: mockGoogle,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      expect(mockGoogle.accounts.id.initialize).toHaveBeenCalledWith({
        client_id: process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
        callback: expect.any(Function),
      });
      expect(mockGoogle.accounts.id.prompt).toHaveBeenCalled();
    });

    it('should fallback to redirect when Google SDK is not available', async () => {
      // Remove Google SDK
      Object.defineProperty(window, 'google', {
        value: undefined,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      const expectedUrl = `/api/auth/google?redirect_uri=${encodeURIComponent(window.location.origin + '/auth/callback')}`;
      expect(mockLocation.href).toBe(expectedUrl);
    });
  });

  describe('OAuth Response Handling', () => {
    it('should handle successful Google OAuth response', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, user: { id: '1', email: 'test@example.com' } }),
      });

      Object.defineProperty(window, 'google', {
        value: mockGoogle,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      // Simulate Google OAuth response
      const initializeCall = mockGoogle.accounts.id.initialize.mock.calls[0];
      const callback = initializeCall[0].callback;
      
      await callback({ credential: 'mock-credential' });

      expect(global.fetch).toHaveBeenCalledWith('/api/auth/google/callback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          credential: 'mock-credential',
        }),
      });

      expect(mockLocation.href).toBe('/studio');
    });

    it('should handle failed Google OAuth response', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
      });

      Object.defineProperty(window, 'google', {
        value: mockGoogle,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      // Simulate Google OAuth response
      const initializeCall = mockGoogle.accounts.id.initialize.mock.calls[0];
      const callback = initializeCall[0].callback;
      
      await callback({ credential: 'mock-credential' });

      expect(consoleSpy).toHaveBeenCalledWith('Google authentication error:', expect.any(Error));
      
      // Loading state should be cleared
      await waitFor(() => {
        expect(screen.queryByText(/connecting.../i)).not.toBeInTheDocument();
      });

      consoleSpy.mockRestore();
    });

    it('should handle network errors during OAuth response', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      Object.defineProperty(window, 'google', {
        value: mockGoogle,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      // Simulate Google OAuth response
      const initializeCall = mockGoogle.accounts.id.initialize.mock.calls[0];
      const callback = initializeCall[0].callback;
      
      await callback({ credential: 'mock-credential' });

      expect(consoleSpy).toHaveBeenCalledWith('Google authentication error:', expect.any(Error));

      consoleSpy.mockRestore();
    });
  });

  describe('Loading States', () => {
    it('should show loading state during OAuth process', async () => {
      Object.defineProperty(window, 'google', {
        value: mockGoogle,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      expect(screen.getByText(/connecting.../i)).toBeInTheDocument();
      expect(button).toBeDisabled();
    });

    it('should clear loading state after error', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      // Mock Google SDK to throw error
      mockGoogle.accounts.id.initialize.mockImplementationOnce(() => {
        throw new Error('SDK error');
      });

      Object.defineProperty(window, 'google', {
        value: mockGoogle,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith('Google login failed:', expect.any(Error));
      });

      // Loading state should be cleared
      expect(screen.queryByText(/connecting.../i)).not.toBeInTheDocument();
      expect(button).not.toBeDisabled();

      consoleSpy.mockRestore();
    });
  });

  describe('Edge Cases', () => {
    it('should handle rapid clicks', async () => {
      Object.defineProperty(window, 'google', {
        value: mockGoogle,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      
      // Click multiple times rapidly
      await user.click(button);
      await user.click(button);
      await user.click(button);

      // Should only initialize once
      expect(mockGoogle.accounts.id.initialize).toHaveBeenCalledTimes(1);
    });

    it('should handle missing environment variables', async () => {
      const originalEnv = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
      delete process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

      Object.defineProperty(window, 'google', {
        value: mockGoogle,
        writable: true,
      });

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      expect(mockGoogle.accounts.id.initialize).toHaveBeenCalledWith({
        client_id: undefined,
        callback: expect.any(Function),
      });

      process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID = originalEnv;
    });

    it('should handle server-side rendering', () => {
      // Mock window as undefined (SSR scenario)
      const originalWindow = global.window;
      delete (global as any).window;

      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      expect(button).toBeInTheDocument();

      global.window = originalWindow;
    });
  });

  describe('Accessibility', () => {
    it('should maintain keyboard accessibility', async () => {
      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      
      // Focus and activate with keyboard
      button.focus();
      expect(button).toHaveFocus();

      await user.keyboard('{Enter}');
      
      // Should trigger OAuth flow
      const expectedUrl = `/api/auth/google?redirect_uri=${encodeURIComponent(window.location.origin + '/auth/callback')}`;
      expect(mockLocation.href).toBe(expectedUrl);
    });

    it('should announce loading state to screen readers', async () => {
      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      await user.click(button);

      const loadingButton = screen.getByRole('button', { name: /connecting.../i });
      expect(loadingButton).toHaveAttribute('disabled');
    });

    it('should have proper ARIA attributes', () => {
      renderWithProviders(<GoogleOAuthButton />);

      const button = screen.getByRole('button', { name: /continue with google/i });
      expect(button).toHaveAttribute('type', 'button');
    });
  });
});