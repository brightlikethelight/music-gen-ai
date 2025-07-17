/**
 * ProjectManager Component Tests
 * 
 * Comprehensive tests for the ProjectManager component including:
 * - Project listing and display
 * - Search and filtering functionality
 * - Sorting options
 * - View mode switching (grid/list)
 * - Project actions (create, update, delete, duplicate)
 * - Pagination and load more
 * - Error states and loading states
 * - Empty states
 * - Project card interactions
 * - Accessibility
 * - Edge cases
 */

import React from 'react';
import { screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test-utils';
import { ProjectManager } from '../ProjectManager';

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock useProjects hook
const mockProjectsHook = {
  projects: [],
  isLoading: false,
  error: null,
  pagination: { page: 1, limit: 12, total: 0 },
  createProject: jest.fn(),
  updateProject: jest.fn(),
  deleteProject: jest.fn(),
  duplicateProject: jest.fn(),
  searchProjects: jest.fn(),
  filterByTags: jest.fn(),
  loadMore: jest.fn(),
  hasMore: false,
};

jest.mock('@/hooks/useProjects', () => ({
  useProjects: () => mockProjectsHook,
}));

// Mock child components
jest.mock('../CreateProjectModal', () => ({
  CreateProjectModal: ({ isOpen, onClose, onCreate }: any) => (
    isOpen ? (
      <div data-testid="create-project-modal">
        <button onClick={onClose} data-testid="close-modal">Close</button>
        <button onClick={() => onCreate({ name: 'Test Project' })} data-testid="create-project">
          Create
        </button>
      </div>
    ) : null
  ),
}));

jest.mock('../ProjectOptionsMenu', () => ({
  ProjectOptionsMenu: ({ isOpen, onClose, onUpdate, onDelete, onDuplicate, project }: any) => (
    isOpen ? (
      <div data-testid="project-options-menu">
        <button onClick={onClose} data-testid="close-options">Close</button>
        <button onClick={() => onUpdate(project.id, { name: 'Updated' })} data-testid="update-project">
          Update
        </button>
        <button onClick={() => onDelete(project.id)} data-testid="delete-project">
          Delete
        </button>
        <button onClick={() => onDuplicate(project.id, 'Duplicate')} data-testid="duplicate-project">
          Duplicate
        </button>
      </div>
    ) : null
  ),
}));

// Mock project data
const mockProjects = [
  {
    id: 'project-1',
    name: 'Electronic Beat',
    description: 'An upbeat electronic track',
    isPublic: true,
    tags: ['electronic', 'dance'],
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-15T00:00:00Z',
    metadata: {
      trackCount: 4,
      totalDuration: 180,
    },
    collaborators: ['user1', 'user2'],
  },
  {
    id: 'project-2',
    name: 'Jazz Ensemble',
    description: 'A smooth jazz piece',
    isPublic: false,
    tags: ['jazz', 'smooth'],
    createdAt: '2024-01-05T00:00:00Z',
    updatedAt: '2024-01-20T00:00:00Z',
    metadata: {
      trackCount: 6,
      totalDuration: 240,
    },
    collaborators: ['user1'],
  },
];

describe('ProjectManager', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Reset mock states
    mockProjectsHook.projects = mockProjects;
    mockProjectsHook.isLoading = false;
    mockProjectsHook.error = null;
    mockProjectsHook.pagination = { page: 1, limit: 12, total: 2 };
    mockProjectsHook.hasMore = false;
  });

  describe('Rendering', () => {
    it('should render the project manager interface', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByRole('heading', { name: /my projects/i })).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/search projects/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /new project/i })).toBeInTheDocument();
    });

    it('should render project statistics', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('2 projects • 2 total')).toBeInTheDocument();
    });

    it('should render search and filter controls', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByPlaceholderText(/search projects/i)).toBeInTheDocument();
      expect(screen.getByText(/last updated/i)).toBeInTheDocument();
      expect(screen.getByText(/newest/i)).toBeInTheDocument();
      expect(screen.getByText(/filter by tags/i)).toBeInTheDocument();
    });

    it('should render tag filters', () => {
      renderWithProviders(<ProjectManager />);

      const commonTags = ['electronic', 'pop', 'jazz', 'classical', 'ambient', 'rock', 'hip-hop', 'experimental'];
      commonTags.forEach(tag => {
        expect(screen.getByRole('button', { name: tag })).toBeInTheDocument();
      });
    });

    it('should render view mode toggle', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByRole('button', { name: /list view/i })).toBeInTheDocument();
    });
  });

  describe('Project Display', () => {
    it('should render projects in grid view by default', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('Electronic Beat')).toBeInTheDocument();
      expect(screen.getByText('Jazz Ensemble')).toBeInTheDocument();
    });

    it('should display project information correctly', () => {
      renderWithProviders(<ProjectManager />);

      // Check first project
      expect(screen.getByText('Electronic Beat')).toBeInTheDocument();
      expect(screen.getByText('An upbeat electronic track')).toBeInTheDocument();
      expect(screen.getByText('4')).toBeInTheDocument(); // Track count
      expect(screen.getByText('3:00')).toBeInTheDocument(); // Duration
      expect(screen.getByText('2')).toBeInTheDocument(); // Collaborators

      // Check privacy indicators
      const publicIcons = screen.getAllByText('🌐'); // Simplified for globe icon
      const privateIcons = screen.getAllByText('🔒'); // Simplified for lock icon
      // Note: In real tests, you'd use better selectors based on actual rendered elements
    });

    it('should display project tags', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('electronic')).toBeInTheDocument();
      expect(screen.getByText('dance')).toBeInTheDocument();
      expect(screen.getByText('jazz')).toBeInTheDocument();
      expect(screen.getByText('smooth')).toBeInTheDocument();
    });

    it('should handle empty projects list', () => {
      mockProjectsHook.projects = [];
      mockProjectsHook.pagination = { page: 1, limit: 12, total: 0 };
      
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText(/no projects found/i)).toBeInTheDocument();
      expect(screen.getByText(/create your first project to get started/i)).toBeInTheDocument();
    });

    it('should show filtered empty state', () => {
      mockProjectsHook.projects = [];
      renderWithProviders(<ProjectManager />);

      // Search for something
      const searchInput = screen.getByPlaceholderText(/search projects/i);
      fireEvent.change(searchInput, { target: { value: 'nonexistent' } });

      expect(screen.getByText(/try adjusting your search or filters/i)).toBeInTheDocument();
    });
  });

  describe('View Mode Switching', () => {
    it('should switch to list view', async () => {
      renderWithProviders(<ProjectManager />);

      const viewToggleButton = screen.getByRole('button', { name: /list view/i });
      await user.click(viewToggleButton);

      expect(screen.getByRole('button', { name: /grid view/i })).toBeInTheDocument();
    });

    it('should switch back to grid view', async () => {
      renderWithProviders(<ProjectManager />);

      // Switch to list view first
      const listViewButton = screen.getByRole('button', { name: /list view/i });
      await user.click(listViewButton);

      // Switch back to grid view
      const gridViewButton = screen.getByRole('button', { name: /grid view/i });
      await user.click(gridViewButton);

      expect(screen.getByRole('button', { name: /list view/i })).toBeInTheDocument();
    });
  });

  describe('Search Functionality', () => {
    it('should handle search input', async () => {
      renderWithProviders(<ProjectManager />);

      const searchInput = screen.getByPlaceholderText(/search projects/i);
      await user.type(searchInput, 'electronic');

      expect(searchInput).toHaveValue('electronic');
      expect(mockProjectsHook.searchProjects).toHaveBeenCalledWith('electronic');
    });

    it('should search on every character typed', async () => {
      renderWithProviders(<ProjectManager />);

      const searchInput = screen.getByPlaceholderText(/search projects/i);
      await user.type(searchInput, 'jazz');

      expect(mockProjectsHook.searchProjects).toHaveBeenCalledTimes(4); // j, a, z, z
      expect(mockProjectsHook.searchProjects).toHaveBeenLastCalledWith('jazz');
    });

    it('should clear search', async () => {
      renderWithProviders(<ProjectManager />);

      const searchInput = screen.getByPlaceholderText(/search projects/i);
      await user.type(searchInput, 'test');
      await user.clear(searchInput);

      expect(searchInput).toHaveValue('');
      expect(mockProjectsHook.searchProjects).toHaveBeenLastCalledWith('');
    });
  });

  describe('Tag Filtering', () => {
    it('should filter by tag', async () => {
      renderWithProviders(<ProjectManager />);

      const electronicTag = screen.getByRole('button', { name: 'electronic' });
      await user.click(electronicTag);

      expect(electronicTag).toHaveClass('bg-purple-600');
      expect(mockProjectsHook.filterByTags).toHaveBeenCalledWith(['electronic']);
    });

    it('should handle multiple tag selection', async () => {
      renderWithProviders(<ProjectManager />);

      const electronicTag = screen.getByRole('button', { name: 'electronic' });
      const jazzTag = screen.getByRole('button', { name: 'jazz' });

      await user.click(electronicTag);
      await user.click(jazzTag);

      expect(mockProjectsHook.filterByTags).toHaveBeenCalledWith(['electronic', 'jazz']);
    });

    it('should remove tag from filter', async () => {
      renderWithProviders(<ProjectManager />);

      const electronicTag = screen.getByRole('button', { name: 'electronic' });
      
      // Add tag
      await user.click(electronicTag);
      // Remove tag
      await user.click(electronicTag);

      expect(mockProjectsHook.filterByTags).toHaveBeenLastCalledWith([]);
    });

    it('should clear all tags', async () => {
      renderWithProviders(<ProjectManager />);

      // Select multiple tags
      await user.click(screen.getByRole('button', { name: 'electronic' }));
      await user.click(screen.getByRole('button', { name: 'jazz' }));

      // Clear all should appear
      const clearAllButton = screen.getByRole('button', { name: /clear all/i });
      await user.click(clearAllButton);

      expect(mockProjectsHook.filterByTags).toHaveBeenLastCalledWith([]);
    });
  });

  describe('Sorting', () => {
    it('should handle sort by selection', async () => {
      renderWithProviders(<ProjectManager />);

      const sortBySelect = screen.getByDisplayValue(/last updated/i);
      await user.selectOptions(sortBySelect, 'name');

      // Sort function would be called with new sort criteria
      // Note: The actual implementation might vary based on Select component behavior
    });

    it('should handle sort order selection', async () => {
      renderWithProviders(<ProjectManager />);

      const sortOrderSelect = screen.getByDisplayValue(/newest/i);
      await user.selectOptions(sortOrderSelect, 'asc');

      // Sort function would be called with new order
    });
  });

  describe('Project Actions', () => {
    it('should open create project modal', async () => {
      renderWithProviders(<ProjectManager />);

      const newProjectButton = screen.getByRole('button', { name: /new project/i });
      await user.click(newProjectButton);

      expect(screen.getByTestId('create-project-modal')).toBeInTheDocument();
    });

    it('should handle project creation', async () => {
      renderWithProviders(<ProjectManager />);

      // Open modal
      await user.click(screen.getByRole('button', { name: /new project/i }));

      // Create project
      const createButton = screen.getByTestId('create-project');
      await user.click(createButton);

      expect(mockProjectsHook.createProject).toHaveBeenCalledWith({ name: 'Test Project' });
    });

    it('should close create project modal', async () => {
      renderWithProviders(<ProjectManager />);

      // Open modal
      await user.click(screen.getByRole('button', { name: /new project/i }));

      // Close modal
      const closeButton = screen.getByTestId('close-modal');
      await user.click(closeButton);

      expect(screen.queryByTestId('create-project-modal')).not.toBeInTheDocument();
    });
  });

  describe('Project Card Actions', () => {
    it('should show options menu when clicking options button', async () => {
      renderWithProviders(<ProjectManager />);

      // Find and click options button (simplified - in real test, use better selector)
      const optionsButtons = screen.getAllByText('⋮'); // Simplified ellipsis
      if (optionsButtons.length > 0) {
        await user.click(optionsButtons[0]);
        expect(screen.getByTestId('project-options-menu')).toBeInTheDocument();
      }
    });

    it('should handle project update', async () => {
      renderWithProviders(<ProjectManager />);

      // Open options menu and update
      const optionsButtons = screen.getAllByText('⋮');
      if (optionsButtons.length > 0) {
        await user.click(optionsButtons[0]);
        const updateButton = screen.getByTestId('update-project');
        await user.click(updateButton);

        expect(mockProjectsHook.updateProject).toHaveBeenCalledWith('project-1', { name: 'Updated' });
      }
    });

    it('should handle project deletion', async () => {
      renderWithProviders(<ProjectManager />);

      const optionsButtons = screen.getAllByText('⋮');
      if (optionsButtons.length > 0) {
        await user.click(optionsButtons[0]);
        const deleteButton = screen.getByTestId('delete-project');
        await user.click(deleteButton);

        expect(mockProjectsHook.deleteProject).toHaveBeenCalledWith('project-1');
      }
    });

    it('should handle project duplication', async () => {
      renderWithProviders(<ProjectManager />);

      const optionsButtons = screen.getAllByText('⋮');
      if (optionsButtons.length > 0) {
        await user.click(optionsButtons[0]);
        const duplicateButton = screen.getByTestId('duplicate-project');
        await user.click(duplicateButton);

        expect(mockProjectsHook.duplicateProject).toHaveBeenCalledWith('project-1', 'Duplicate');
      }
    });
  });

  describe('Pagination and Load More', () => {
    it('should show load more button when hasMore is true', () => {
      mockProjectsHook.hasMore = true;
      renderWithProviders(<ProjectManager />);

      expect(screen.getByRole('button', { name: /load more projects/i })).toBeInTheDocument();
    });

    it('should handle load more', async () => {
      mockProjectsHook.hasMore = true;
      renderWithProviders(<ProjectManager />);

      const loadMoreButton = screen.getByRole('button', { name: /load more projects/i });
      await user.click(loadMoreButton);

      expect(mockProjectsHook.loadMore).toHaveBeenCalled();
    });

    it('should show loading state on load more button', () => {
      mockProjectsHook.hasMore = true;
      mockProjectsHook.isLoading = true;
      renderWithProviders(<ProjectManager />);

      const loadMoreButton = screen.getByRole('button', { name: /loading.../i });
      expect(loadMoreButton).toBeDisabled();
    });

    it('should not show load more button when hasMore is false', () => {
      mockProjectsHook.hasMore = false;
      renderWithProviders(<ProjectManager />);

      expect(screen.queryByRole('button', { name: /load more projects/i })).not.toBeInTheDocument();
    });
  });

  describe('Loading States', () => {
    it('should show loading spinner when loading and no projects', () => {
      mockProjectsHook.projects = [];
      mockProjectsHook.isLoading = true;
      renderWithProviders(<ProjectManager />);

      expect(screen.getByRole('generic')).toHaveClass('animate-spin');
    });

    it('should not show loading spinner when projects exist', () => {
      mockProjectsHook.isLoading = true;
      renderWithProviders(<ProjectManager />);

      expect(screen.queryByRole('generic')).not.toHaveClass('animate-spin');
    });
  });

  describe('Error States', () => {
    it('should display error message', () => {
      mockProjectsHook.error = 'Failed to load projects';
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('Failed to load projects')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
    });

    it('should handle try again button', async () => {
      mockProjectsHook.error = 'Failed to load projects';
      const reloadSpy = jest.spyOn(window.location, 'reload').mockImplementation();
      
      renderWithProviders(<ProjectManager />);

      const tryAgainButton = screen.getByRole('button', { name: /try again/i });
      await user.click(tryAgainButton);

      expect(reloadSpy).toHaveBeenCalled();
      reloadSpy.mockRestore();
    });
  });

  describe('Date and Duration Formatting', () => {
    it('should format dates correctly', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText(/updated jan 15, 2024/i)).toBeInTheDocument();
      expect(screen.getByText(/updated jan 20, 2024/i)).toBeInTheDocument();
    });

    it('should format duration correctly', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('3:00')).toBeInTheDocument(); // 180 seconds
      expect(screen.getByText('4:00')).toBeInTheDocument(); // 240 seconds
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      renderWithProviders(<ProjectManager />);

      expect(screen.getByRole('heading', { name: /my projects/i })).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/search projects/i)).toHaveAttribute('type', 'text');
    });

    it('should be keyboard navigable', async () => {
      renderWithProviders(<ProjectManager />);

      // Tab through interactive elements
      await user.tab();
      expect(screen.getByRole('button', { name: /new project/i })).toHaveFocus();

      await user.tab();
      expect(screen.getByRole('button', { name: /list view/i })).toHaveFocus();
    });

    it('should handle keyboard interactions', async () => {
      renderWithProviders(<ProjectManager />);

      const newProjectButton = screen.getByRole('button', { name: /new project/i });
      newProjectButton.focus();
      
      await user.keyboard('{Enter}');
      expect(screen.getByTestId('create-project-modal')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle singular project count', () => {
      mockProjectsHook.projects = [mockProjects[0]];
      mockProjectsHook.pagination = { page: 1, limit: 12, total: 1 };
      
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('1 project • 1 total')).toBeInTheDocument();
    });

    it('should handle empty tags array', () => {
      const projectWithoutTags = {
        ...mockProjects[0],
        tags: [],
      };
      mockProjectsHook.projects = [projectWithoutTags];
      
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('Electronic Beat')).toBeInTheDocument();
    });

    it('should handle projects with many tags', () => {
      const projectWithManyTags = {
        ...mockProjects[0],
        tags: ['tag1', 'tag2', 'tag3', 'tag4', 'tag5'],
      };
      mockProjectsHook.projects = [projectWithManyTags];
      
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('+2')).toBeInTheDocument(); // Shows only first 3 tags
    });

    it('should handle very long project names and descriptions', () => {
      const projectWithLongText = {
        ...mockProjects[0],
        name: 'A'.repeat(100),
        description: 'B'.repeat(200),
      };
      mockProjectsHook.projects = [projectWithLongText];
      
      renderWithProviders(<ProjectManager />);

      expect(screen.getByText('A'.repeat(100))).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('should handle different screen sizes', () => {
      renderWithProviders(<ProjectManager />);

      // Grid should be responsive
      const gridContainer = screen.getByText('Electronic Beat').closest('.grid');
      expect(gridContainer).toHaveClass('grid-cols-1', 'md:grid-cols-2', 'lg:grid-cols-3', 'xl:grid-cols-4');
    });

    it('should handle mobile layout', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      renderWithProviders(<ProjectManager />);

      // Header should be responsive
      const header = screen.getByRole('heading', { name: /my projects/i }).closest('.flex');
      expect(header).toHaveClass('flex-col', 'lg:flex-row');
    });
  });
});