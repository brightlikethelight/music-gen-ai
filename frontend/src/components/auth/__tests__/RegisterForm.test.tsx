/**
 * RegisterForm Component Tests
 * 
 * Comprehensive tests for the RegisterForm component including:
 * - Form validation
 * - Password strength checking
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
import { RegisterForm } from '../RegisterForm';

// Mock useRouter from next/navigation
const mockPush = jest.fn();
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock password service
const mockPasswordService = {
  checkPasswordStrength: jest.fn(),
};

jest.mock('@/lib/auth', () => ({
  passwordService: mockPasswordService,
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
}));

describe('RegisterForm', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    mockPush.mockClear();
    mockPasswordService.checkPasswordStrength.mockReturnValue({
      score: 5,
      feedback: [],
      isStrong: true,
    });
  });

  describe('Rendering', () => {
    it('should render registration form with all required fields', () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByRole('heading', { name: /create account/i })).toBeInTheDocument();
      expect(screen.getByLabelText(/first name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/last name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/^password/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
    });

    it('should render terms and marketing checkboxes', () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByRole('checkbox', { name: /terms of service/i })).toBeInTheDocument();
      expect(screen.getByRole('checkbox', { name: /marketing communications/i })).toBeInTheDocument();
    });

    it('should render social login options', () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByText(/or continue with/i)).toBeInTheDocument();
    });

    it('should render link to login page', () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByRole('link', { name: /already have an account\? sign in/i })).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('should show validation errors for empty required fields', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const submitButton = screen.getByRole('button', { name: /create account/i });
      expect(submitButton).toBeDisabled();

      // Fill in required fields one by one to test validation
      const emailInput = screen.getByLabelText(/email address/i);
      await user.type(emailInput, 'test@example.com');
      expect(submitButton).toBeDisabled();

      const usernameInput = screen.getByLabelText(/username/i);
      await user.type(usernameInput, 'testuser');
      expect(submitButton).toBeDisabled();

      const passwordInput = screen.getByLabelText(/^password/i);
      await user.type(passwordInput, 'StrongPassword123!');
      expect(submitButton).toBeDisabled();

      const termsCheckbox = screen.getByRole('checkbox', { name: /terms of service/i });
      await user.click(termsCheckbox);
      expect(submitButton).not.toBeDisabled();
    });

    it('should show validation error for invalid email format', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const emailInput = screen.getByLabelText(/email address/i);
      await user.type(emailInput, 'invalid-email');

      // HTML5 validation should trigger
      expect(emailInput).toHaveAttribute('type', 'email');
      expect(emailInput).toBeInvalid();
    });

    it('should show password mismatch error', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const passwordInput = screen.getByLabelText(/^password/i);
      const confirmPasswordInput = screen.getByLabelText(/confirm password/i);

      await user.type(passwordInput, 'StrongPassword123!');
      await user.type(confirmPasswordInput, 'DifferentPassword123!');

      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
    });

    it('should show password match indicator when passwords match', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const passwordInput = screen.getByLabelText(/^password/i);
      const confirmPasswordInput = screen.getByLabelText(/confirm password/i);

      await user.type(passwordInput, 'StrongPassword123!');
      await user.type(confirmPasswordInput, 'StrongPassword123!');

      expect(screen.getByText(/passwords match/i)).toBeInTheDocument();
    });

    it('should show error when terms are not accepted', async () => {
      const mockRegister = jest.fn();
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill form without accepting terms
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');

      // Try to submit without accepting terms
      const submitButton = screen.getByRole('button', { name: /create account/i });
      expect(submitButton).toBeDisabled();
    });

    it('should clear error message when user starts typing', async () => {
      const mockRegister = jest.fn().mockRejectedValue(new Error('Registration failed'));
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill form and submit to trigger error
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/registration failed/i)).toBeInTheDocument();
      });

      // Clear error by typing
      await user.clear(screen.getByLabelText(/email address/i));
      await user.type(screen.getByLabelText(/email address/i), 'new@example.com');

      await waitFor(() => {
        expect(screen.queryByText(/registration failed/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Password Strength', () => {
    it('should show password strength indicator', async () => {
      mockPasswordService.checkPasswordStrength.mockReturnValue({
        score: 3,
        feedback: ['Add more special characters'],
        isStrong: false,
      });

      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const passwordInput = screen.getByLabelText(/^password/i);
      await user.type(passwordInput, 'weakpass');

      expect(screen.getByText(/medium/i)).toBeInTheDocument();
      expect(screen.getByText(/add more special characters/i)).toBeInTheDocument();
    });

    it('should show strong password indicator', async () => {
      mockPasswordService.checkPasswordStrength.mockReturnValue({
        score: 6,
        feedback: [],
        isStrong: true,
      });

      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const passwordInput = screen.getByLabelText(/^password/i);
      await user.type(passwordInput, 'VeryStrongPassword123!@#');

      expect(screen.getByText(/strong/i)).toBeInTheDocument();
    });

    it('should disable submit button for weak passwords', async () => {
      mockPasswordService.checkPasswordStrength.mockReturnValue({
        score: 2,
        feedback: ['Password is too weak'],
        isStrong: false,
      });

      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      // Fill all fields
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'weak');
      await user.type(screen.getByLabelText(/confirm password/i), 'weak');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));

      const submitButton = screen.getByRole('button', { name: /create account/i });
      expect(submitButton).toBeDisabled();
    });
  });

  describe('User Interactions', () => {
    it('should handle successful registration', async () => {
      const mockRegister = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill form
      await user.type(screen.getByLabelText(/first name/i), 'John');
      await user.type(screen.getByLabelText(/last name/i), 'Doe');
      await user.type(screen.getByLabelText(/email address/i), 'john@example.com');
      await user.type(screen.getByLabelText(/username/i), 'johndoe');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('checkbox', { name: /marketing communications/i }));

      const submitButton = screen.getByRole('button', { name: /create account/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockRegister).toHaveBeenCalledWith({
          email: 'john@example.com',
          username: 'johndoe',
          password: 'StrongPassword123!',
          firstName: 'John',
          lastName: 'Doe',
        });
      });

      expect(mockPush).toHaveBeenCalledWith('/auth/verify-email?email=john%40example.com');
    });

    it('should handle registration with only required fields', async () => {
      const mockRegister = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill only required fields
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));

      const submitButton = screen.getByRole('button', { name: /create account/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockRegister).toHaveBeenCalledWith({
          email: 'test@example.com',
          username: 'testuser',
          password: 'StrongPassword123!',
          firstName: undefined,
          lastName: undefined,
        });
      });
    });

    it('should toggle password visibility', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const passwordInput = screen.getByLabelText(/^password/i);
      const passwordToggle = screen.getByRole('button', { name: /toggle password visibility/i });

      expect(passwordInput).toHaveAttribute('type', 'password');

      await user.click(passwordToggle);
      expect(passwordInput).toHaveAttribute('type', 'text');

      await user.click(passwordToggle);
      expect(passwordInput).toHaveAttribute('type', 'password');
    });

    it('should toggle confirm password visibility', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const confirmPasswordInput = screen.getByLabelText(/confirm password/i);
      const confirmPasswordToggle = screen.getAllByRole('button', { name: /toggle password visibility/i })[1];

      expect(confirmPasswordInput).toHaveAttribute('type', 'password');

      await user.click(confirmPasswordToggle);
      expect(confirmPasswordInput).toHaveAttribute('type', 'text');

      await user.click(confirmPasswordToggle);
      expect(confirmPasswordInput).toHaveAttribute('type', 'password');
    });

    it('should handle form submission with Enter key', async () => {
      const mockRegister = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill form
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      
      // Submit with Enter key
      await user.keyboard('{Enter}');

      await waitFor(() => {
        expect(mockRegister).toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    it('should display error message for registration failure', async () => {
      const mockRegister = jest.fn().mockRejectedValue(new Error('Registration failed'));
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill and submit form
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/an unexpected error occurred/i)).toBeInTheDocument();
      });
    });

    it('should display error message for existing email', async () => {
      const mockRegister = jest.fn().mockResolvedValue({ 
        success: false, 
        error: 'Email already exists' 
      });
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill and submit form
      await user.type(screen.getByLabelText(/email address/i), 'existing@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/email already exists/i)).toBeInTheDocument();
      });
    });

    it('should handle network errors', async () => {
      simulateNetworkError('/api/auth/register');
      
      const mockRegister = jest.fn().mockRejectedValue(new Error('Network error'));
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill and submit form
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/an unexpected error occurred/i)).toBeInTheDocument();
      });
    });

    it('should prevent form submission when passwords do not match', async () => {
      const mockRegister = jest.fn();
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill form with mismatched passwords
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'DifferentPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));

      const submitButton = screen.getByRole('button', { name: /create account/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
      });

      expect(mockRegister).not.toHaveBeenCalled();
    });

    it('should prevent form submission when password is weak', async () => {
      mockPasswordService.checkPasswordStrength.mockReturnValue({
        score: 2,
        feedback: ['Password is too weak'],
        isStrong: false,
      });

      const mockRegister = jest.fn();
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill form with weak password
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'weak');
      await user.type(screen.getByLabelText(/confirm password/i), 'weak');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));

      const submitButton = screen.getByRole('button', { name: /create account/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/please choose a stronger password/i)).toBeInTheDocument();
      });

      expect(mockRegister).not.toHaveBeenCalled();
    });
  });

  describe('Loading States', () => {
    it('should show loading state during registration', async () => {
      const mockRegister = jest.fn(() => new Promise(resolve => setTimeout(resolve, 100)));
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill and submit form
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('button', { name: /create account/i }));

      expect(screen.getByText(/creating account.../i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /creating account.../i })).toBeDisabled();

      // Wait for loading to complete
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
      });
    });

    it('should disable form fields during loading', async () => {
      const mockRegister = jest.fn(() => new Promise(resolve => setTimeout(resolve, 100)));
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill and submit form
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('button', { name: /create account/i }));

      // Check if fields are disabled during loading
      expect(screen.getByLabelText(/email address/i)).toBeDisabled();
      expect(screen.getByLabelText(/username/i)).toBeDisabled();
      expect(screen.getByLabelText(/^password/i)).toBeDisabled();
      expect(screen.getByLabelText(/confirm password/i)).toBeDisabled();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels and roles', () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByLabelText(/email address/i)).toHaveAttribute('aria-required', 'true');
      expect(screen.getByLabelText(/username/i)).toHaveAttribute('aria-required', 'true');
      expect(screen.getByLabelText(/^password/i)).toHaveAttribute('aria-required', 'true');
      expect(screen.getByLabelText(/confirm password/i)).toHaveAttribute('aria-required', 'true');
    });

    it('should be keyboard navigable', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      // Tab through all interactive elements
      await user.tab();
      expect(screen.getByLabelText(/first name/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/last name/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/email address/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/username/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/^password/i)).toHaveFocus();

      await user.tab();
      expect(screen.getAllByRole('button', { name: /toggle password visibility/i })[0]).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/confirm password/i)).toHaveFocus();
    });

    it('should have proper autocomplete attributes', () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      expect(screen.getByLabelText(/first name/i)).toHaveAttribute('autoComplete', 'given-name');
      expect(screen.getByLabelText(/last name/i)).toHaveAttribute('autoComplete', 'family-name');
      expect(screen.getByLabelText(/email address/i)).toHaveAttribute('autoComplete', 'email');
      expect(screen.getByLabelText(/username/i)).toHaveAttribute('autoComplete', 'username');
      expect(screen.getByLabelText(/^password/i)).toHaveAttribute('autoComplete', 'new-password');
      expect(screen.getByLabelText(/confirm password/i)).toHaveAttribute('autoComplete', 'new-password');
    });

    it('should announce error messages to screen readers', async () => {
      const mockRegister = jest.fn().mockRejectedValue(new Error('Registration failed'));
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill and submit form to trigger error
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        const errorMessage = screen.getByText(/an unexpected error occurred/i);
        expect(errorMessage.closest('[role="alert"]')).toBeInTheDocument();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle extremely long input values', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const longText = 'a'.repeat(1000);
      const emailInput = screen.getByLabelText(/email address/i);
      const usernameInput = screen.getByLabelText(/username/i);

      await user.type(emailInput, longText + '@example.com');
      await user.type(usernameInput, longText);

      expect(emailInput).toHaveValue(longText + '@example.com');
      expect(usernameInput).toHaveValue(longText);
    });

    it('should handle special characters in input', async () => {
      renderWithProviders(<RegisterForm />, {
        authValue: mockUnauthenticatedContextValue,
      });

      const specialEmail = 'test+special@example.com';
      const specialUsername = 'user_name-123';
      const specialPassword = 'p@$$w0rd!@#$%^&*()';

      await user.type(screen.getByLabelText(/email address/i), specialEmail);
      await user.type(screen.getByLabelText(/username/i), specialUsername);
      await user.type(screen.getByLabelText(/^password/i), specialPassword);

      expect(screen.getByLabelText(/email address/i)).toHaveValue(specialEmail);
      expect(screen.getByLabelText(/username/i)).toHaveValue(specialUsername);
      expect(screen.getByLabelText(/^password/i)).toHaveValue(specialPassword);
    });

    it('should handle rapid form submissions', async () => {
      const mockRegister = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill form
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));

      const submitButton = screen.getByRole('button', { name: /create account/i });

      // Rapidly click submit multiple times
      await user.click(submitButton);
      await user.click(submitButton);
      await user.click(submitButton);

      // Should only call register once
      await waitFor(() => {
        expect(mockRegister).toHaveBeenCalledTimes(1);
      });
    });

    it('should handle empty name fields gracefully', async () => {
      const mockRegister = jest.fn().mockResolvedValue({ success: true });
      renderWithProviders(<RegisterForm />, {
        authValue: { ...mockUnauthenticatedContextValue, register: mockRegister },
      });

      // Fill form without name fields
      await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
      await user.type(screen.getByLabelText(/username/i), 'testuser');
      await user.type(screen.getByLabelText(/^password/i), 'StrongPassword123!');
      await user.type(screen.getByLabelText(/confirm password/i), 'StrongPassword123!');
      await user.click(screen.getByRole('checkbox', { name: /terms of service/i }));
      await user.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(mockRegister).toHaveBeenCalledWith({
          email: 'test@example.com',
          username: 'testuser',
          password: 'StrongPassword123!',
          firstName: undefined,
          lastName: undefined,
        });
      });
    });
  });
});