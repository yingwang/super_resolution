/**
 * Main Application Logic
 * Ties together the video processor and enhancer modules
 */

class App {
    constructor() {
        // DOM elements
        this.elements = {
            // Video elements
            originalVideo: document.getElementById('originalVideo'),
            enhancedCanvas: document.getElementById('enhancedCanvas'),

            // Source controls
            sourceRadios: document.querySelectorAll('input[name="source"]'),
            urlInput: document.getElementById('urlInput'),
            videoUrlField: document.getElementById('videoUrl'),
            loadUrlBtn: document.getElementById('loadUrl'),
            sampleVideos: document.getElementById('sampleVideos'),
            sampleSelect: document.getElementById('sampleSelect'),
            loadSampleBtn: document.getElementById('loadSample'),

            // Enhancement controls
            enableEnhancement: document.getElementById('enableEnhancement'),
            enhancementMode: document.getElementById('enhancementMode'),
            intensity: document.getElementById('intensity'),
            intensityValue: document.getElementById('intensityValue'),
            brightness: document.getElementById('brightness'),
            brightnessValue: document.getElementById('brightnessValue'),
            contrast: document.getElementById('contrast'),
            contrastValue: document.getElementById('contrastValue'),
            saturation: document.getElementById('saturation'),
            saturationValue: document.getElementById('saturationValue'),

            // Playback controls
            playPauseBtn: document.getElementById('playPause'),
            stopBtn: document.getElementById('stop'),
            fullscreenBtn: document.getElementById('fullscreen'),
            currentTime: document.getElementById('currentTime'),
            duration: document.getElementById('duration'),
            seekBar: document.getElementById('seekBar'),
            showComparison: document.getElementById('showComparison'),

            // Info displays
            originalInfo: document.getElementById('originalInfo'),
            enhancedInfo: document.getElementById('enhancedInfo'),

            // Stats
            fpsDisplay: document.getElementById('fps'),
            frameTimeDisplay: document.getElementById('frameTime'),
            resolutionDisplay: document.getElementById('resolution'),
            backendDisplay: document.getElementById('backend')
        };

        // Modules
        this.enhancer = new VideoEnhancer();
        this.processor = new VideoProcessor();

        // State
        this.currentSource = 'sample';
        this.videoLoaded = false;

        // Bind methods
        this._bindEvents();
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            // Show loading state
            this._showNotification('Initializing...', 'info');

            // Initialize enhancer
            const backend = await this.enhancer.init(this.elements.enhancedCanvas);
            console.log('Enhancer initialized with backend:', backend);

            // Initialize processor
            await this.processor.init(
                this.elements.originalVideo,
                this.elements.enhancedCanvas,
                this.enhancer
            );

            // Set up stats callback
            this.processor.onStats = (stats) => this._updateStats(stats);
            this.processor.onError = (error) => this._showNotification(error.message, 'error');

            // Update backend display
            const backendInfo = this.enhancer.getBackendInfo();
            this.elements.backendDisplay.textContent = `${backendInfo.enhancer} / ${backendInfo.tensorflow}`;

            // Load default sample video
            await this._loadSampleVideo();

            this._showNotification('Ready! Click Play to start.', 'success');

        } catch (error) {
            console.error('Initialization error:', error);
            this._showNotification(`Initialization failed: ${error.message}`, 'error');
        }
    }

    /**
     * Bind event listeners
     */
    _bindEvents() {
        // Source selection
        this.elements.sourceRadios.forEach(radio => {
            radio.addEventListener('change', (e) => this._onSourceChange(e.target.value));
        });

        // URL loading
        this.elements.loadUrlBtn.addEventListener('click', () => this._loadFromUrl());
        this.elements.videoUrlField.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this._loadFromUrl();
        });

        // Sample video loading
        this.elements.loadSampleBtn.addEventListener('click', () => this._loadSampleVideo());

        // Enhancement controls
        this.elements.enableEnhancement.addEventListener('change', () => this._updateEnhancementSettings());
        this.elements.enhancementMode.addEventListener('change', () => this._updateEnhancementSettings());

        // Slider controls
        const sliders = ['intensity', 'brightness', 'contrast', 'saturation'];
        sliders.forEach(name => {
            this.elements[name].addEventListener('input', (e) => {
                this.elements[`${name}Value`].textContent = e.target.value;
                this._updateEnhancementSettings();
            });
        });

        // Playback controls
        this.elements.playPauseBtn.addEventListener('click', () => this._togglePlayPause());
        this.elements.stopBtn.addEventListener('click', () => this._stop());
        this.elements.fullscreenBtn.addEventListener('click', () => this._toggleFullscreen());
        this.elements.seekBar.addEventListener('input', (e) => this._seek(e.target.value));
        this.elements.showComparison.addEventListener('change', (e) => this._toggleComparison(e.target.checked));

        // Video time update
        this.elements.originalVideo.addEventListener('timeupdate', () => this._updateTimeDisplay());
        this.elements.originalVideo.addEventListener('loadedmetadata', () => this._onVideoMetadata());
        this.elements.originalVideo.addEventListener('play', () => this._onPlay());
        this.elements.originalVideo.addEventListener('pause', () => this._onPause());
        this.elements.originalVideo.addEventListener('ended', () => this._onEnded());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this._handleKeyboard(e));
    }

    /**
     * Handle source type change
     */
    _onSourceChange(source) {
        this.currentSource = source;

        // Update UI visibility
        this.elements.urlInput.style.display = source === 'url' ? 'flex' : 'none';
        this.elements.sampleVideos.style.display = source === 'sample' ? 'flex' : 'none';

        // If webcam selected, load webcam immediately
        if (source === 'webcam') {
            this._loadFromWebcam();
        }
    }

    /**
     * Load video from URL
     */
    async _loadFromUrl() {
        const url = this.elements.videoUrlField.value.trim();
        if (!url) {
            this._showNotification('Please enter a video URL', 'error');
            return;
        }

        try {
            this._showNotification('Loading video...', 'info');
            const info = await this.processor.loadFromUrl(url);
            this._onVideoLoaded(info);
        } catch (error) {
            this._showNotification(`Failed to load video: ${error.message}`, 'error');
        }
    }

    /**
     * Load sample video
     */
    async _loadSampleVideo() {
        const url = this.elements.sampleSelect.value;

        try {
            this._showNotification('Loading sample video...', 'info');
            const info = await this.processor.loadFromUrl(url);
            this._onVideoLoaded(info);
        } catch (error) {
            this._showNotification(`Failed to load sample: ${error.message}`, 'error');
        }
    }

    /**
     * Load from webcam
     */
    async _loadFromWebcam() {
        try {
            this._showNotification('Accessing webcam...', 'info');
            const info = await this.processor.loadFromWebcam();
            this._onVideoLoaded(info);

            // Auto-play webcam
            await this.processor.play();
        } catch (error) {
            this._showNotification(error.message, 'error');
            // Revert to sample source
            document.querySelector('input[name="source"][value="sample"]').checked = true;
            this._onSourceChange('sample');
        }
    }

    /**
     * Called when video is loaded
     */
    _onVideoLoaded(info) {
        this.videoLoaded = true;

        // Enable controls
        this.elements.playPauseBtn.disabled = false;
        this.elements.stopBtn.disabled = false;
        this.elements.seekBar.disabled = !isFinite(info.duration);

        // Update info displays
        this.elements.originalInfo.textContent = `${info.width}x${info.height}`;
        this.elements.resolutionDisplay.textContent = `${info.width}x${info.height}`;

        if (isFinite(info.duration)) {
            this.elements.duration.textContent = this._formatTime(info.duration);
        } else {
            this.elements.duration.textContent = 'LIVE';
        }

        this._showNotification('Video loaded successfully!', 'success');
    }

    /**
     * Video metadata loaded
     */
    _onVideoMetadata() {
        const video = this.elements.originalVideo;
        this.elements.originalInfo.textContent = `${video.videoWidth}x${video.videoHeight}`;
        this.elements.resolutionDisplay.textContent = `${video.videoWidth}x${video.videoHeight}`;

        if (isFinite(video.duration)) {
            this.elements.duration.textContent = this._formatTime(video.duration);
            this.elements.seekBar.max = video.duration;
        }
    }

    /**
     * Toggle play/pause
     */
    async _togglePlayPause() {
        if (!this.videoLoaded) return;

        try {
            if (this.elements.originalVideo.paused) {
                await this.processor.play();
            } else {
                this.processor.pause();
            }
        } catch (error) {
            this._showNotification(error.message, 'error');
        }
    }

    /**
     * Stop playback
     */
    _stop() {
        this.processor.stop();
    }

    /**
     * Play event handler
     */
    _onPlay() {
        this.elements.playPauseBtn.textContent = '⏸️ Pause';
    }

    /**
     * Pause event handler
     */
    _onPause() {
        this.elements.playPauseBtn.textContent = '▶️ Play';
    }

    /**
     * Video ended handler
     */
    _onEnded() {
        this.elements.playPauseBtn.textContent = '▶️ Play';
    }

    /**
     * Seek to position
     */
    _seek(value) {
        this.processor.seek(parseFloat(value));
    }

    /**
     * Update time display
     */
    _updateTimeDisplay() {
        const video = this.elements.originalVideo;
        this.elements.currentTime.textContent = this._formatTime(video.currentTime);

        if (isFinite(video.duration)) {
            this.elements.seekBar.value = video.currentTime;
        }
    }

    /**
     * Format time in MM:SS
     */
    _formatTime(seconds) {
        if (!isFinite(seconds)) return '--:--';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    /**
     * Toggle fullscreen
     */
    _toggleFullscreen() {
        const container = document.querySelector('.video-container');

        if (!document.fullscreenElement) {
            container.requestFullscreen().catch(err => {
                this._showNotification(`Fullscreen error: ${err.message}`, 'error');
            });
        } else {
            document.exitFullscreen();
        }
    }

    /**
     * Toggle comparison view
     */
    _toggleComparison(enabled) {
        document.querySelector('.main-content').classList.toggle('split-view', enabled);
    }

    /**
     * Update enhancement settings
     */
    _updateEnhancementSettings() {
        const enabled = this.elements.enableEnhancement.checked;

        const settings = {
            mode: enabled ? this.elements.enhancementMode.value : 'passthrough',
            intensity: this.elements.intensity.value / 100,
            brightness: this.elements.brightness.value / 100,
            contrast: this.elements.contrast.value / 100,
            saturation: this.elements.saturation.value / 100
        };

        this.enhancer.updateSettings(settings);

        // Update enhanced info display
        const mode = enabled ? this.elements.enhancementMode.options[this.elements.enhancementMode.selectedIndex].text : 'Off';
        const video = this.elements.originalVideo;

        if (settings.mode === 'superres') {
            this.elements.enhancedInfo.textContent = `${video.videoWidth * 2}x${video.videoHeight * 2} (${mode})`;
        } else {
            this.elements.enhancedInfo.textContent = `${video.videoWidth}x${video.videoHeight} (${mode})`;
        }
    }

    /**
     * Update performance stats display
     */
    _updateStats(stats) {
        this.elements.fpsDisplay.textContent = `${stats.fps} fps`;
        this.elements.frameTimeDisplay.textContent = `${stats.frameTime} ms`;
    }

    /**
     * Handle keyboard shortcuts
     */
    _handleKeyboard(e) {
        // Ignore if typing in input
        if (e.target.tagName === 'INPUT') return;

        switch (e.key) {
            case ' ':
            case 'k':
                e.preventDefault();
                this._togglePlayPause();
                break;
            case 'f':
                e.preventDefault();
                this._toggleFullscreen();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                this.processor.seek(this.elements.originalVideo.currentTime - 5);
                break;
            case 'ArrowRight':
                e.preventDefault();
                this.processor.seek(this.elements.originalVideo.currentTime + 5);
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.elements.intensity.value = Math.min(100, parseInt(this.elements.intensity.value) + 10);
                this.elements.intensityValue.textContent = this.elements.intensity.value;
                this._updateEnhancementSettings();
                break;
            case 'ArrowDown':
                e.preventDefault();
                this.elements.intensity.value = Math.max(0, parseInt(this.elements.intensity.value) - 10);
                this.elements.intensityValue.textContent = this.elements.intensity.value;
                this._updateEnhancementSettings();
                break;
        }
    }

    /**
     * Show notification
     */
    _showNotification(message, type = 'info') {
        // Remove existing notifications
        document.querySelectorAll('.notification').forEach(n => n.remove());

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    /**
     * Cleanup resources
     */
    dispose() {
        this.processor.dispose();
        this.enhancer.dispose();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
    window.app.init();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.app) {
        window.app.dispose();
    }
});
