/**
 * MusicGen Web UI - Interactive JavaScript
 * Handles music generation, progress tracking, and result display
 */

class MusicGenUI {
    constructor() {
        this.apiBase = '/api';
        this.currentJobId = null;
        this.pollInterval = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadSavedSettings();
    }

    initializeElements() {
        // Form elements
        this.promptInput = document.getElementById('prompt');
        this.durationInput = document.getElementById('duration');
        this.temperatureInput = document.getElementById('temperature');
        this.guidanceInput = document.getElementById('guidance');
        this.formatSelect = document.getElementById('format');
        this.generateBtn = document.getElementById('generateBtn');
        
        // Display elements
        this.tempValue = document.getElementById('tempValue');
        this.guidanceValue = document.getElementById('guidanceValue');
        
        // Sections
        this.progressSection = document.getElementById('progressSection');
        this.resultSection = document.getElementById('resultSection');
        this.errorSection = document.getElementById('errorSection');
        
        // Progress elements
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        
        // Result elements
        this.audioPlayer = document.getElementById('audioPlayer');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.regenerateBtn = document.getElementById('regenerateBtn');
        
        // Error elements
        this.errorMessage = document.getElementById('errorMessage');
        this.retryBtn = document.getElementById('retryBtn');
    }

    attachEventListeners() {
        // Generate button
        this.generateBtn.addEventListener('click', () => this.generateMusic());
        
        // Sliders
        this.temperatureInput.addEventListener('input', (e) => {
            this.tempValue.textContent = e.target.value;
            this.saveSettings();
        });
        
        this.guidanceInput.addEventListener('input', (e) => {
            this.guidanceValue.textContent = e.target.value;
            this.saveSettings();
        });
        
        // Result actions
        this.downloadBtn.addEventListener('click', () => this.downloadAudio());
        this.regenerateBtn.addEventListener('click', () => this.resetUI());
        
        // Error retry
        this.retryBtn.addEventListener('click', () => this.resetUI());
        
        // Enter key on prompt
        this.promptInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.generateMusic();
            }
        });
    }

    loadSavedSettings() {
        // Load saved preferences from localStorage
        const saved = localStorage.getItem('musicgen_settings');
        if (saved) {
            try {
                const settings = JSON.parse(saved);
                if (settings.temperature) {
                    this.temperatureInput.value = settings.temperature;
                    this.tempValue.textContent = settings.temperature;
                }
                if (settings.guidance) {
                    this.guidanceInput.value = settings.guidance;
                    this.guidanceValue.textContent = settings.guidance;
                }
                if (settings.format) {
                    this.formatSelect.value = settings.format;
                }
            } catch (e) {
                console.error('Failed to load settings:', e);
            }
        }
    }

    saveSettings() {
        const settings = {
            temperature: this.temperatureInput.value,
            guidance: this.guidanceInput.value,
            format: this.formatSelect.value
        };
        localStorage.setItem('musicgen_settings', JSON.stringify(settings));
    }

    async generateMusic() {
        // Validate input
        const prompt = this.promptInput.value.trim();
        if (!prompt) {
            this.showError('Please enter a music description');
            return;
        }

        // Disable form
        this.setFormEnabled(false);
        
        // Hide other sections and show progress
        this.hideAllSections();
        this.progressSection.classList.remove('hidden');
        this.updateProgress(0, 'Sending request...');

        try {
            // Send generation request
            const response = await fetch(`${this.apiBase}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    duration: parseFloat(this.durationInput.value),
                    temperature: parseFloat(this.temperatureInput.value),
                    guidance_scale: parseFloat(this.guidanceInput.value),
                    format: this.formatSelect.value
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Generation failed');
            }

            const data = await response.json();
            this.currentJobId = data.job_id;
            
            // Start polling for status
            this.updateProgress(10, 'Request accepted, waiting for generation...');
            this.startPolling();
            
        } catch (error) {
            this.showError(error.message);
            this.setFormEnabled(true);
        }
    }

    startPolling() {
        // Poll every 2 seconds for job status
        this.pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`${this.apiBase}/status/${this.currentJobId}`);
                
                if (!response.ok) {
                    throw new Error('Failed to get status');
                }
                
                const status = await response.json();
                
                if (status.status === 'processing') {
                    const progress = status.progress || 20;
                    this.updateProgress(progress, 'Generating your music...');
                    
                } else if (status.status === 'completed') {
                    this.stopPolling();
                    this.updateProgress(100, 'Complete!');
                    
                    // Show result after a brief delay
                    setTimeout(() => {
                        this.showResult(status.result_url);
                    }, 500);
                    
                } else if (status.status === 'failed') {
                    this.stopPolling();
                    this.showError(status.error || 'Generation failed');
                }
                
            } catch (error) {
                this.stopPolling();
                this.showError('Lost connection to server');
            }
        }, 2000);
    }

    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    updateProgress(percent, message) {
        this.progressFill.style.width = `${percent}%`;
        this.progressText.textContent = message;
    }

    showResult(audioUrl) {
        this.hideAllSections();
        this.resultSection.classList.remove('hidden');
        
        // Set audio source
        this.audioPlayer.src = audioUrl;
        this.audioPlayer.load();
        
        // Store URL for download
        this.currentAudioUrl = audioUrl;
        
        // Re-enable form
        this.setFormEnabled(true);
    }

    showError(message) {
        this.hideAllSections();
        this.errorSection.classList.remove('hidden');
        this.errorMessage.textContent = message;
        this.setFormEnabled(true);
    }

    downloadAudio() {
        if (this.currentAudioUrl) {
            const link = document.createElement('a');
            link.href = this.currentAudioUrl;
            link.download = `musicgen_${Date.now()}.${this.formatSelect.value}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    resetUI() {
        this.hideAllSections();
        this.setFormEnabled(true);
        this.currentJobId = null;
        this.currentAudioUrl = null;
        this.stopPolling();
    }

    hideAllSections() {
        this.progressSection.classList.add('hidden');
        this.resultSection.classList.add('hidden');
        this.errorSection.classList.add('hidden');
    }

    setFormEnabled(enabled) {
        this.generateBtn.disabled = !enabled;
        this.promptInput.disabled = !enabled;
        this.durationInput.disabled = !enabled;
        this.temperatureInput.disabled = !enabled;
        this.guidanceInput.disabled = !enabled;
        this.formatSelect.disabled = !enabled;
        
        if (enabled) {
            this.generateBtn.textContent = 'Generate Music';
        } else {
            this.generateBtn.textContent = 'Generating...';
            this.generateBtn.classList.add('loading');
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.musicGenUI = new MusicGenUI();
    
    // Add example prompts functionality
    const examplePrompts = [
        "peaceful acoustic guitar with soft piano, ambient background",
        "upbeat electronic dance music with heavy bass and synths",
        "cinematic orchestral piece with strings and brass, epic mood",
        "jazz quartet with saxophone, piano, bass, and drums",
        "lo-fi hip hop beat with vinyl crackle and mellow vibes"
    ];
    
    // Add click-to-fill for example prompts (if we add them to the UI)
    document.querySelectorAll('.example-prompt').forEach(el => {
        el.addEventListener('click', () => {
            window.musicGenUI.promptInput.value = el.textContent;
        });
    });
});

// Handle page visibility changes to stop polling when hidden
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.musicGenUI) {
        window.musicGenUI.stopPolling();
    }
});