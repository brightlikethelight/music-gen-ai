/**
 * LoginForm Component Tests
 * 
 * Comprehensive tests for the LoginForm component including:
 * - Form validation
 * - User interactions
 * - Success and error states
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, mockAuthContextValue, mockUnauthenticatedContextValue } from '../../../test-utils';
import { server, simulateNetworkError, simulateServerError } from '../../../test-utils/server';
import { rest } from 'msw';
import LoginForm from '../LoginForm';

// Mock useNavigate from react-router-dom
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

describe('LoginForm', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    mockNavigate.mockClear();
  });

  describe('Rendering', () => {
    it('should render login form with all required fields', () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByRole('heading', { name: /sign in/i })).toBeInTheDocument();
      expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
      expect(screen.getByRole('checkbox', { name: /remember me/i })).toBeInTheDocument();
    });

    it('should render social login options', () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByRole('button', { name: /continue with google/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /continue with github/i })).toBeInTheDocument();
    });

    it('should render links to registration and forgot password', () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByRole('link', { name: /don't have an account\? sign up/i })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /forgot your password\?/i })).toBeInTheDocument();
    });

    it('should redirect if user is already authenticated', () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockAuthContextValue,
      });

      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });
  });

  describe('Form Validation', () => {
    it('should show validation errors for empty fields', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/email is required/i)).toBeInTheDocument();
        expect(screen.getByText(/password is required/i)).toBeInTheDocument();
      });
    });

    it('should show validation error for invalid email format', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const emailInput = screen.getByLabelText(/email/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'invalid-email');
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/please enter a valid email address/i)).toBeInTheDocument();
      });
    });

    it('should show validation error for short password', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, '123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/password must be at least 8 characters/i)).toBeInTheDocument();
      });
    });

    it('should clear validation errors when user types', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const emailInput = screen.getByLabelText(/email/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      // Trigger validation error
      await user.click(submitButton);
      await waitFor(() => {
        expect(screen.getByText(/email is required/i)).toBeInTheDocument();
      });

      // Start typing to clear error
      await user.type(emailInput, 'test@example.com');
      await waitFor(() => {
        expect(screen.queryByText(/email is required/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('User Interactions', () => {
    it('should handle successful login', async () => {
      const mockLogin = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith({
          email: 'test@example.com',
          password: 'password123',
          rememberMe: false,
        });
      });

      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });

    it('should handle remember me checkbox', async () => {
      const mockLogin = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const rememberMeCheckbox = screen.getByRole('checkbox', { name: /remember me/i });
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.click(rememberMeCheckbox);
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith({
          email: 'test@example.com',
          password: 'password123',
          rememberMe: true,
        });
      });
    });

    it('should toggle password visibility', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const passwordInput = screen.getByLabelText(/password/i);
      const toggleButton = screen.getByRole('button', { name: /toggle password visibility/i });

      expect(passwordInput).toHaveAttribute('type', 'password');

      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute('type', 'text');

      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute('type', 'password');
    });

    it('should handle form submission with Enter key', async () => {
      const mockLogin = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.keyboard('{Enter}');

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    it('should display error message for invalid credentials', async () => {
      const mockLogin = jest.fn().mockRejectedValue(new Error('Invalid credentials'));
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'wrong@example.com');
      await user.type(passwordInput, 'wrongpassword');
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });

      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('should display error message for network errors', async () => {
      simulateNetworkError('/api/auth/login');
      
      const mockLogin = jest.fn().mockRejectedValue(new Error('Network error'));
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/network error/i)).toBeInTheDocument();
      });
    });

    it('should display error message for rate limiting', async () => {
      server.use(
        rest.post('/api/auth/login', (req, res, ctx) => {
          return res(
            ctx.status(429),
            ctx.json({ detail: 'Too many login attempts. Please try again later.' })
          );
        })
      );

      const mockLogin = jest.fn().mockRejectedValue(new Error('Too many login attempts'));
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/too many login attempts/i)).toBeInTheDocument();
      });
    });

    it('should clear error message when user starts typing', async () => {
      const mockLogin = jest.fn().mockRejectedValue(new Error('Invalid credentials'));
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      // Trigger error
      await user.type(emailInput, 'wrong@example.com');
      await user.type(passwordInput, 'wrongpassword');
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });

      // Clear error by typing
      await user.clear(emailInput);
      await user.type(emailInput, 'test@example.com');

      await waitFor(() => {
        expect(screen.queryByText(/invalid credentials/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Loading States', () => {
    it('should show loading state during login', async () => {
      const mockLogin = jest.fn(() => new Promise(resolve => setTimeout(resolve, 100)));
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      expect(screen.getByRole('button', { name: /signing in.../i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /signing in.../i })).toBeDisabled();

      // Wait for loading to complete
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
      });
    });

    it('should disable form fields during loading', async () => {
      const mockLogin = jest.fn(() => new Promise(resolve => setTimeout(resolve, 100)));
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);

      expect(emailInput).toBeDisabled();
      expect(passwordInput).toBeDisabled();
      expect(submitButton).toBeDisabled();
    });
  });

  describe('Social Login', () => {
    it('should handle Google OAuth login', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const googleButton = screen.getByRole('button', { name: /continue with google/i });
      
      // Mock window.open
      const mockOpen = jest.fn();
      Object.defineProperty(window, 'open', { value: mockOpen });

      await user.click(googleButton);

      expect(mockOpen).toHaveBeenCalledWith(
        '/api/auth/google',
        'popup',
        expect.stringContaining('width=500,height=600')
      );
    });

    it('should handle GitHub OAuth login', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const githubButton = screen.getByRole('button', { name: /continue with github/i });
      
      // Mock window.open
      const mockOpen = jest.fn();
      Object.defineProperty(window, 'open', { value: mockOpen });

      await user.click(githubButton);

      expect(mockOpen).toHaveBeenCalledWith(
        '/api/auth/github',
        'popup',
        expect.stringContaining('width=500,height=600')
      );
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels and roles', () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByRole('form', { name: /login form/i })).toBeInTheDocument();
      expect(screen.getByLabelText(/email/i)).toHaveAttribute('aria-required', 'true');
      expect(screen.getByLabelText(/password/i)).toHaveAttribute('aria-required', 'true');
    });

    it('should associate error messages with form fields', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        const emailInput = screen.getByLabelText(/email/i);
        const emailError = screen.getByText(/email is required/i);
        
        expect(emailInput).toHaveAttribute('aria-describedby', expect.stringContaining(emailError.id));
        expect(emailInput).toHaveAttribute('aria-invalid', 'true');
      });
    });

    it('should be keyboard navigable', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      // Tab through all interactive elements
      await user.tab();
      expect(screen.getByLabelText(/email/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/password/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /toggle password visibility/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('checkbox', { name: /remember me/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /sign in/i })).toHaveFocus();
    });

    it('should announce error messages to screen readers', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const submitButton = screen.getByRole('button', { name: /sign in/i });
      await user.click(submitButton);

      await waitFor(() => {
        const errorMessage = screen.getByText(/email is required/i);
        expect(errorMessage).toHaveAttribute('role', 'alert');
        expect(errorMessage).toHaveAttribute('aria-live', 'polite');
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle form submission when already logged in', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockAuthContextValue,
      });

      // Should redirect immediately
      expect(mockNavigate).toHaveBeenCalledWith('/dashboard');
    });

    it('should handle extremely long input values', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);

      const longEmail = 'a'.repeat(1000) + '@example.com';
      const longPassword = 'a'.repeat(1000);

      await user.type(emailInput, longEmail);
      await user.type(passwordInput, longPassword);

      expect(emailInput).toHaveValue(longEmail);
      expect(passwordInput).toHaveValue(longPassword);
    });

    it('should handle special characters in input', async () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);

      const specialEmail = 'test+special@example.com';
      const specialPassword = 'p@$$w0rd!@#$%^&*()';

      await user.type(emailInput, specialEmail);
      await user.type(passwordInput, specialPassword);

      expect(emailInput).toHaveValue(specialEmail);
      expect(passwordInput).toHaveValue(specialPassword);
    });

    it('should handle browser autofill', () => {
      renderWithProviders(<LoginForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);

      expect(emailInput).toHaveAttribute('autoComplete', 'email');
      expect(passwordInput).toHaveAttribute('autoComplete', 'current-password');
    });

    it('should handle rapid form submissions', async () => {
      const mockLogin = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<LoginForm />, {
        authValue: { ...mockUnauthenticatedContextValue, login: mockLogin },
      });

      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole('button', { name: /sign in/i });

      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');

      // Rapidly click submit multiple times
      await user.click(submitButton);
      await user.click(submitButton);
      await user.click(submitButton);

      // Should only call login once
      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledTimes(1);
      });
    });
  });
});