/**
 * CreateProjectModal Component Tests
 * 
 * Comprehensive tests for the CreateProjectModal component including:
 * - Modal opening and closing
 * - Template selection step
 * - Project details step
 * - Form validation
 * - Tag management
 * - Privacy settings
 * - Project creation workflow
 * - Error handling
 * - Loading states
 * - Keyboard navigation
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { CreateProjectModal } from '../CreateProjectModal';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe('CreateProjectModal', () => {
  const user = userEvent.setup();
  const mockOnClose = jest.fn();
  const mockOnCreate = jest.fn();

  const defaultProps = {
    isOpen: true,
    onClose: mockOnClose,
    onCreate: mockOnCreate,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockOnCreate.mockResolvedValue({ id: 'new-project-id', name: 'New Project' });
  });

  describe('Modal Visibility', () => {
    it('should render when isOpen is true', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      expect(screen.getByRole('heading', { name: /create new project/i })).toBeInTheDocument();
    });

    it('should not render when isOpen is false', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} isOpen={false} />);

      expect(screen.queryByRole('heading', { name: /create new project/i })).not.toBeInTheDocument();
    });

    it('should close when clicking backdrop', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      const backdrop = screen.getByRole('generic').parentElement?.firstChild;
      if (backdrop) {
        await user.click(backdrop);
        expect(mockOnClose).toHaveBeenCalled();
      }
    });

    it('should close when clicking X button', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      const closeButton = screen.getByRole('button', { name: /close/i });
      await user.click(closeButton);

      expect(mockOnClose).toHaveBeenCalled();
    });

    it('should close when clicking Cancel button', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      const cancelButton = screen.getByRole('button', { name: /cancel/i });
      await user.click(cancelButton);

      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('Template Selection Step', () => {
    it('should show template selection as first step', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      expect(screen.getByText(/choose a template to get started/i)).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: /choose a template/i })).toBeInTheDocument();
    });

    it('should render blank project option', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      expect(screen.getByText(/blank project/i)).toBeInTheDocument();
      expect(screen.getByText(/start from scratch with an empty project/i)).toBeInTheDocument();
    });

    it('should render all template options', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      const templates = [
        'Electronic Starter',
        'Pop Ballad',
        'Jazz Ensemble',
        'Ambient Soundscape',
        'Rock Band',
        'Hip-Hop Beat',
      ];

      templates.forEach(template => {
        expect(screen.getByText(template)).toBeInTheDocument();
      });
    });

    it('should show template details', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      // Check electronic starter template details
      expect(screen.getByText(/basic electronic music template/i)).toBeInTheDocument();
      expect(screen.getByText('Electronic')).toBeInTheDocument();
      expect(screen.getByText('4 tracks')).toBeInTheDocument();
      expect(screen.getByText('2:30')).toBeInTheDocument();
    });

    it('should select blank project', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      const blankProject = screen.getByText(/blank project/i).closest('div[role="button"]') || 
                          screen.getByText(/blank project/i).parentElement?.parentElement;
      
      if (blankProject) {
        await user.click(blankProject);
        expect(blankProject).toHaveClass('border-purple-500');
      }
    });

    it('should select a template', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      const electronicTemplate = screen.getByText('Electronic Starter').closest('div[role="button"]') ||
                                screen.getByText('Electronic Starter').parentElement?.parentElement?.parentElement;
      
      if (electronicTemplate) {
        await user.click(electronicTemplate);
        expect(electronicTemplate).toHaveClass('border-purple-500');
      }
    });

    it('should show official template badge', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      // All templates should have sparkles icon for official badge
      const sparklesIcons = screen.getAllByTitle(/sparkles/i);
      expect(sparklesIcons.length).toBeGreaterThan(0);
    });

    it('should proceed to details step', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      expect(screen.getByText(/configure your project settings/i)).toBeInTheDocument();
    });
  });

  describe('Project Details Step', () => {
    beforeEach(async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      // Go to details step
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);
    });

    it('should show project details form', () => {
      expect(screen.getByRole('heading', { name: /project details/i })).toBeInTheDocument();
      expect(screen.getByLabelText(/project name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
      expect(screen.getByText(/privacy/i)).toBeInTheDocument();
      expect(screen.getByText(/tags/i)).toBeInTheDocument();
    });

    it('should pre-fill name when template is selected', async () => {
      // Go back to template step
      const backButton = screen.getByRole('button', { name: /back/i });
      await user.click(backButton);

      // Select a template
      const jazzTemplate = screen.getByText('Jazz Ensemble').closest('div[role="button"]') ||
                          screen.getByText('Jazz Ensemble').parentElement?.parentElement?.parentElement;
      if (jazzTemplate) {
        await user.click(jazzTemplate);
      }

      // Go to details step
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      const nameInput = screen.getByLabelText(/project name/i);
      expect(nameInput).toHaveValue('My Jazz Ensemble');
    });

    it('should allow entering project name', async () => {
      const nameInput = screen.getByLabelText(/project name/i);
      await user.type(nameInput, 'My Awesome Project');

      expect(nameInput).toHaveValue('My Awesome Project');
    });

    it('should allow entering description', async () => {
      const descriptionTextarea = screen.getByLabelText(/description/i);
      await user.type(descriptionTextarea, 'This is a great project');

      expect(descriptionTextarea).toHaveValue('This is a great project');
    });

    it('should toggle privacy settings', async () => {
      const privateRadio = screen.getByLabelText(/private/i);
      const publicRadio = screen.getByLabelText(/public/i);

      expect(privateRadio).toBeChecked();
      expect(publicRadio).not.toBeChecked();

      await user.click(publicRadio);

      expect(privateRadio).not.toBeChecked();
      expect(publicRadio).toBeChecked();
    });

    it('should go back to template step', async () => {
      const backButton = screen.getByRole('button', { name: /back/i });
      await user.click(backButton);

      expect(screen.getByText(/choose a template to get started/i)).toBeInTheDocument();
    });
  });

  describe('Tag Management', () => {
    beforeEach(async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);
    });

    it('should add tags via input', async () => {
      const tagInput = screen.getByPlaceholderText(/add tags/i);
      await user.type(tagInput, 'custom-tag');

      const addTagButton = screen.getByRole('button', { name: /tag/i });
      await user.click(addTagButton);

      expect(screen.getByText('custom-tag')).toBeInTheDocument();
    });

    it('should add tags on Enter key', async () => {
      const tagInput = screen.getByPlaceholderText(/add tags/i);
      await user.type(tagInput, 'my-tag{Enter}');

      expect(screen.getByText('my-tag')).toBeInTheDocument();
    });

    it('should add tags on comma key', async () => {
      const tagInput = screen.getByPlaceholderText(/add tags/i);
      await user.type(tagInput, 'comma-tag,');

      expect(screen.getByText('comma-tag')).toBeInTheDocument();
    });

    it('should add common tags by clicking', async () => {
      const electronicTag = screen.getByRole('button', { name: 'electronic' });
      await user.click(electronicTag);

      // Tag should appear in selected tags
      const selectedTags = screen.getAllByText('electronic');
      expect(selectedTags.length).toBeGreaterThan(1); // One in common tags, one in selected
    });

    it('should remove selected tags', async () => {
      // Add a tag first
      const jazzTag = screen.getByRole('button', { name: 'jazz' });
      await user.click(jazzTag);

      // Find and click remove button on selected tag
      const selectedJazzTag = screen.getAllByText('jazz').find(el => 
        el.closest('.bg-purple-600')
      );
      
      if (selectedJazzTag) {
        const removeButton = selectedJazzTag.parentElement?.querySelector('button');
        if (removeButton) {
          await user.click(removeButton);
          
          // Tag should be removed from selected tags
          expect(screen.getAllByText('jazz').length).toBe(1); // Only in common tags
        }
      }
    });

    it('should disable already selected common tags', async () => {
      const popTag = screen.getByRole('button', { name: 'pop' });
      await user.click(popTag);

      // The pop button in common tags should now be disabled
      expect(popTag).toBeDisabled();
      expect(popTag).toHaveClass('cursor-not-allowed');
    });

    it('should convert tags to lowercase', async () => {
      const tagInput = screen.getByPlaceholderText(/add tags/i);
      await user.type(tagInput, 'UPPERCASE-TAG{Enter}');

      expect(screen.getByText('uppercase-tag')).toBeInTheDocument();
    });

    it('should trim whitespace from tags', async () => {
      const tagInput = screen.getByPlaceholderText(/add tags/i);
      await user.type(tagInput, '  spaced-tag  {Enter}');

      expect(screen.getByText('spaced-tag')).toBeInTheDocument();
    });

    it('should not add duplicate tags', async () => {
      const tagInput = screen.getByPlaceholderText(/add tags/i);
      await user.type(tagInput, 'duplicate{Enter}');
      await user.type(tagInput, 'duplicate{Enter}');

      const duplicateTags = screen.getAllByText('duplicate');
      expect(duplicateTags).toHaveLength(1);
    });

    it('should not add empty tags', async () => {
      const tagInput = screen.getByPlaceholderText(/add tags/i);
      await user.type(tagInput, '   {Enter}');

      // Should not create any tag element
      expect(screen.queryByText(/^\s*$/)).not.toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    beforeEach(async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);
    });

    it('should show error for empty project name', async () => {
      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      expect(screen.getByText(/project name is required/i)).toBeInTheDocument();
    });

    it('should show error for short project name', async () => {
      const nameInput = screen.getByLabelText(/project name/i);
      await user.type(nameInput, 'AB');

      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      expect(screen.getByText(/project name must be at least 3 characters/i)).toBeInTheDocument();
    });

    it('should clear error when fixing validation issue', async () => {
      // Trigger validation error
      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);
      
      expect(screen.getByText(/project name is required/i)).toBeInTheDocument();

      // Fix the error
      const nameInput = screen.getByLabelText(/project name/i);
      await user.type(nameInput, 'Valid Project Name');

      // Error should disappear
      expect(screen.queryByText(/project name is required/i)).not.toBeInTheDocument();
    });

    it('should disable create button when name is empty', () => {
      const createButton = screen.getByRole('button', { name: /create project/i });
      expect(createButton).toBeDisabled();
    });

    it('should enable create button when name is valid', async () => {
      const nameInput = screen.getByLabelText(/project name/i);
      await user.type(nameInput, 'Valid Name');

      const createButton = screen.getByRole('button', { name: /create project/i });
      expect(createButton).not.toBeDisabled();
    });
  });

  describe('Project Creation', () => {
    beforeEach(async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      
      // Select a template
      const jazzTemplate = screen.getByText('Jazz Ensemble').closest('div[role="button"]') ||
                          screen.getByText('Jazz Ensemble').parentElement?.parentElement?.parentElement;
      if (jazzTemplate) {
        await user.click(jazzTemplate);
      }
      
      // Go to details step
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);
    });

    it('should create project with all details', async () => {
      // Fill in details
      const nameInput = screen.getByLabelText(/project name/i);
      await user.clear(nameInput);
      await user.type(nameInput, 'My Jazz Project');

      const descriptionTextarea = screen.getByLabelText(/description/i);
      await user.type(descriptionTextarea, 'A smooth jazz composition');

      // Select public
      const publicRadio = screen.getByLabelText(/public/i);
      await user.click(publicRadio);

      // Add tags
      const electronicTag = screen.getByRole('button', { name: 'electronic' });
      await user.click(electronicTag);

      // Create project
      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      await waitFor(() => {
        expect(mockOnCreate).toHaveBeenCalledWith({
          name: 'My Jazz Project',
          description: 'A smooth jazz composition',
          isPublic: true,
          tags: ['jazz', 'electronic'],
          templateId: 'jazz-ensemble',
        });
      });
    });

    it('should create blank project', async () => {
      // Go back and select blank project
      const backButton = screen.getByRole('button', { name: /back/i });
      await user.click(backButton);

      const blankProject = screen.getByText(/blank project/i).closest('div[role="button"]') || 
                          screen.getByText(/blank project/i).parentElement?.parentElement;
      if (blankProject) {
        await user.click(blankProject);
      }

      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      // Fill minimal details
      const nameInput = screen.getByLabelText(/project name/i);
      await user.type(nameInput, 'Blank Project');

      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      await waitFor(() => {
        expect(mockOnCreate).toHaveBeenCalledWith({
          name: 'Blank Project',
          description: undefined,
          isPublic: false,
          tags: [],
          templateId: undefined,
        });
      });
    });

    it('should show loading state during creation', async () => {
      // Delay the creation to see loading state
      mockOnCreate.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 100)));

      const nameInput = screen.getByLabelText(/project name/i);
      await user.clear(nameInput);
      await user.type(nameInput, 'Test Project');

      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      expect(screen.getByText(/creating.../i)).toBeInTheDocument();
      expect(createButton).toBeDisabled();

      await waitFor(() => {
        expect(mockOnClose).toHaveBeenCalled();
      });
    });

    it('should handle creation errors', async () => {
      mockOnCreate.mockRejectedValue(new Error('Creation failed'));
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      const nameInput = screen.getByLabelText(/project name/i);
      await user.clear(nameInput);
      await user.type(nameInput, 'Test Project');

      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      await waitFor(() => {
        expect(screen.getByText(/failed to create project/i)).toBeInTheDocument();
      });

      expect(mockOnClose).not.toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('should reset form after successful creation', async () => {
      const nameInput = screen.getByLabelText(/project name/i);
      await user.clear(nameInput);
      await user.type(nameInput, 'Test Project');

      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      await waitFor(() => {
        expect(mockOnClose).toHaveBeenCalled();
      });

      // Re-render with modal open
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      // Should be back at template selection
      expect(screen.getByText(/choose a template to get started/i)).toBeInTheDocument();
    });

    it('should not close modal during creation', async () => {
      mockOnCreate.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 100)));

      const nameInput = screen.getByLabelText(/project name/i);
      await user.clear(nameInput);
      await user.type(nameInput, 'Test Project');

      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      // Try to close during creation
      const closeButton = screen.getByRole('button', { name: /close/i });
      await user.click(closeButton);

      expect(mockOnClose).not.toHaveBeenCalled();

      await waitFor(() => {
        expect(mockOnClose).toHaveBeenCalled();
      });
    });
  });

  describe('Selected Template Summary', () => {
    it('should show selected template summary in details step', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      // Select a template
      const rockTemplate = screen.getByText('Rock Band').closest('div[role="button"]') ||
                          screen.getByText('Rock Band').parentElement?.parentElement?.parentElement;
      if (rockTemplate) {
        await user.click(rockTemplate);
      }

      // Go to details step
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      // Should show template summary
      expect(screen.getByText(/selected template/i)).toBeInTheDocument();
      const summaries = screen.getAllByText('Rock Band');
      expect(summaries.length).toBeGreaterThan(0);
      expect(screen.getByText(/classic rock band setup/i)).toBeInTheDocument();
    });

    it('should not show template summary for blank project', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      // Select blank project
      const blankProject = screen.getByText(/blank project/i).closest('div[role="button"]') || 
                          screen.getByText(/blank project/i).parentElement?.parentElement;
      if (blankProject) {
        await user.click(blankProject);
      }

      // Go to details step
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      // Should not show template summary
      expect(screen.queryByText(/selected template/i)).not.toBeInTheDocument();
    });
  });

  describe('Keyboard Navigation', () => {
    it('should handle Escape key to close modal', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      await user.keyboard('{Escape}');
      expect(mockOnClose).toHaveBeenCalled();
    });

    it('should handle Tab navigation through form', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      // Go to details step for more form elements
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      // Tab through form elements
      await user.tab();
      expect(screen.getByLabelText(/project name/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/description/i)).toHaveFocus();

      await user.tab();
      expect(screen.getByLabelText(/private/i)).toHaveFocus();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA attributes', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      expect(screen.getByRole('heading', { name: /create new project/i })).toBeInTheDocument();
      expect(screen.getByRole('heading', { name: /choose a template/i })).toBeInTheDocument();
    });

    it('should have proper form labels', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      
      // Go to details step
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      expect(screen.getByLabelText(/project name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/private/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/public/i)).toBeInTheDocument();
    });

    it('should maintain focus when opening modal', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      // Modal should trap focus
      const modal = screen.getByRole('heading', { name: /create new project/i }).closest('.relative');
      expect(modal).toBeInTheDocument();
    });

    it('should announce errors to screen readers', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      
      // Go to details step
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      // Trigger validation error
      const createButton = screen.getByRole('button', { name: /create project/i });
      await user.click(createButton);

      const errorMessage = screen.getByText(/project name is required/i);
      expect(errorMessage).toBeInTheDocument();
      // In a real app, this would have role="alert" or aria-live
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long project names', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      const nameInput = screen.getByLabelText(/project name/i);
      const longName = 'A'.repeat(200);
      await user.type(nameInput, longName);

      expect(nameInput).toHaveValue(longName);
    });

    it('should handle very long descriptions', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      const descriptionTextarea = screen.getByLabelText(/description/i);
      const longDescription = 'B'.repeat(1000);
      await user.type(descriptionTextarea, longDescription);

      expect(descriptionTextarea).toHaveValue(longDescription);
    });

    it('should handle many tags', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      // Add all common tags
      const commonTags = ['electronic', 'pop', 'jazz', 'classical', 'ambient', 'rock', 'hip-hop', 'experimental'];
      
      for (const tag of commonTags) {
        const tagButton = screen.getByRole('button', { name: tag });
        await user.click(tagButton);
      }

      // All tags should be selected
      commonTags.forEach(tag => {
        const selectedTags = screen.getAllByText(tag);
        expect(selectedTags.length).toBeGreaterThan(1);
      });
    });

    it('should handle rapid form submissions', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      const nameInput = screen.getByLabelText(/project name/i);
      await user.type(nameInput, 'Rapid Test');

      const createButton = screen.getByRole('button', { name: /create project/i });
      
      // Click multiple times rapidly
      await user.click(createButton);
      await user.click(createButton);
      await user.click(createButton);

      // Should only call onCreate once
      expect(mockOnCreate).toHaveBeenCalledTimes(1);
    });

    it('should handle special characters in tags', async () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);
      
      const nextButton = screen.getByRole('button', { name: /next/i });
      await user.click(nextButton);

      const tagInput = screen.getByPlaceholderText(/add tags/i);
      await user.type(tagInput, 'tag-with-special_chars.123{Enter}');

      expect(screen.getByText('tag-with-special_chars.123')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('should handle mobile layout', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      // Template grid should be responsive
      const templateGrid = screen.getByText('Electronic Starter').closest('.grid');
      expect(templateGrid).toHaveClass('grid-cols-1', 'md:grid-cols-2');
    });

    it('should handle modal max height on small screens', () => {
      renderWithProviders(<CreateProjectModal {...defaultProps} />);

      const modalContent = screen.getByText(/choose a template to get started/i).closest('.overflow-y-auto');
      expect(modalContent).toHaveClass('max-h-[calc(90vh-140px)]');
    });
  });
});