/**
 * Video Compression Comparison App
 * Compares original video with WebRTC-compressed video side by side
 */

class App {
    constructor() {
        // DOM elements
        this.elements = {
            // Video elements
            originalVideo: document.getElementById('originalVideo'),
            compressedVideo: document.getElementById('compressedVideo'),

            // Source controls
            sourceRadios: document.querySelectorAll('input[name="source"]'),
            urlInput: document.getElementById('urlInput'),
            videoUrlField: document.getElementById('videoUrl'),
            loadUrlBtn: document.getElementById('loadUrl'),
            sampleVideos: document.getElementById('sampleVideos'),
            sampleSelect: document.getElementById('sampleSelect'),
            loadSampleBtn: document.getElementById('loadSample'),

            // Compression controls
            enableCompression: document.getElementById('enableCompression'),
            codecSelect: document.getElementById('codecSelect'),
            bitrateSelect: document.getElementById('bitrateSelect'),
            bitrateSlider: document.getElementById('bitrateSlider'),
            bitrateValue: document.getElementById('bitrateValue'),

            // Playback controls
            playPauseBtn: document.getElementById('playPause'),
            stopBtn: document.getElementById('stop'),
            fullscreenBtn: document.getElementById('fullscreen'),
            currentTime: document.getElementById('currentTime'),
            duration: document.getElementById('duration'),
            seekBar: document.getElementById('seekBar'),

            // Info displays
            originalInfo: document.getElementById('originalInfo'),
            compressedInfo: document.getElementById('compressedInfo'),

            // Stats
            statCodec: document.getElementById('statCodec'),
            statTargetBitrate: document.getElementById('statTargetBitrate'),
            statActualBitrate: document.getElementById('statActualBitrate'),
            statResolution: document.getElementById('statResolution'),
            statFps: document.getElementById('statFps'),
            statPacketsLost: document.getElementById('statPacketsLost'),
            statBytesSent: document.getElementById('statBytesSent'),
            statFramesDropped: document.getElementById('statFramesDropped')
        };

        // Modules
        this.compressionLoopback = new CompressionLoopback();

        // State
        this.currentSource = 'sample';
        this.videoLoaded = false;
        this.mediaStream = null;
        this.canvasStream = null;

        // Bind methods
        this._bindEvents();
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            this._showNotification('Initializing...', 'info');

            // Set up compression loopback callbacks
            this.compressionLoopback.onStats = (stats) => this._updateCompressionStats(stats);
            this.compressionLoopback.onCompressedStream = (stream) => {
                this.elements.compressedVideo.srcObject = stream;
            };

            // Configure video elements
            this.elements.originalVideo.setAttribute('playsinline', '');
            this.elements.originalVideo.setAttribute('muted', '');
            this.elements.originalVideo.muted = true;

            // Load default sample video
            await this._loadSampleVideo();

            this._showNotification('Ready! Click Play to start comparison.', 'success');

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

        // Compression controls
        this.elements.enableCompression.addEventListener('change', () => this._updateCompressionSettings());
        this.elements.codecSelect.addEventListener('change', () => this._updateCompressionSettings());

        // Bitrate controls
        this.elements.bitrateSelect.addEventListener('change', (e) => {
            this.elements.bitrateSlider.value = e.target.value;
            this.elements.bitrateValue.textContent = e.target.value;
            this._updateCompressionSettings();
        });

        this.elements.bitrateSlider.addEventListener('input', (e) => {
            this.elements.bitrateValue.textContent = e.target.value;
            // Update dropdown to closest value
            const options = Array.from(this.elements.bitrateSelect.options);
            const closest = options.reduce((prev, curr) => {
                return Math.abs(curr.value - e.target.value) < Math.abs(prev.value - e.target.value) ? curr : prev;
            });
            this.elements.bitrateSelect.value = closest.value;
            this._updateCompressionSettings();
        });

        // Playback controls
        this.elements.playPauseBtn.addEventListener('click', () => this._togglePlayPause());
        this.elements.stopBtn.addEventListener('click', () => this._stop());
        this.elements.fullscreenBtn.addEventListener('click', () => this._toggleFullscreen());
        this.elements.seekBar.addEventListener('input', (e) => this._seek(e.target.value));

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

        // Load source immediately for webcam/screen
        if (source === 'webcam') {
            this._loadFromWebcam();
        } else if (source === 'screen') {
            this._loadFromScreen();
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
            await this._loadVideo(url);
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
            await this._loadVideo(url);
        } catch (error) {
            this._showNotification(`Failed to load sample: ${error.message}`, 'error');
        }
    }

    /**
     * Load video from URL
     */
    async _loadVideo(url) {
        this._cleanup();

        return new Promise((resolve, reject) => {
            this.elements.originalVideo.src = url;
            this.elements.originalVideo.crossOrigin = 'anonymous';

            this.elements.originalVideo.onloadeddata = () => {
                console.log('Video loaded:', url);
                this._onVideoLoaded({
                    width: this.elements.originalVideo.videoWidth,
                    height: this.elements.originalVideo.videoHeight,
                    duration: this.elements.originalVideo.duration,
                    isLive: false
                });
                resolve();
            };

            this.elements.originalVideo.onerror = (e) => {
                reject(new Error('Failed to load video'));
            };

            this.elements.originalVideo.load();
        });
    }

    /**
     * Load from webcam
     */
    async _loadFromWebcam() {
        try {
            this._cleanup();
            this._showNotification('Accessing webcam...', 'info');

            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });

            this.elements.originalVideo.srcObject = this.mediaStream;

            const track = this.mediaStream.getVideoTracks()[0];
            const settings = track.getSettings();

            this._onVideoLoaded({
                width: settings.width,
                height: settings.height,
                duration: Infinity,
                isLive: true
            });

            // Auto-play for live sources
            await this.elements.originalVideo.play();
            await this._startCompression();

        } catch (error) {
            this._showNotification(error.message || 'Failed to access webcam', 'error');
            // Revert to sample source
            document.querySelector('input[name="source"][value="sample"]').checked = true;
            this._onSourceChange('sample');
        }
    }

    /**
     * Load from screen capture
     */
    async _loadFromScreen() {
        try {
            this._cleanup();
            this._showNotification('Starting screen capture...', 'info');

            this.mediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    cursor: 'always',
                    displaySurface: 'monitor'
                },
                audio: false
            });

            this.elements.originalVideo.srcObject = this.mediaStream;

            // Handle stream ended
            this.mediaStream.getVideoTracks()[0].addEventListener('ended', () => {
                this._onEnded();
            });

            const track = this.mediaStream.getVideoTracks()[0];
            const settings = track.getSettings();

            this._onVideoLoaded({
                width: settings.width,
                height: settings.height,
                duration: Infinity,
                isLive: true
            });

            // Auto-play for live sources
            await this.elements.originalVideo.play();
            await this._startCompression();

        } catch (error) {
            this._showNotification(error.message || 'Screen sharing was cancelled', 'error');
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

        if (isFinite(info.duration)) {
            this.elements.duration.textContent = this._formatTime(info.duration);
        } else {
            this.elements.duration.textContent = 'LIVE';
        }

        // Update target bitrate display
        this.elements.statTargetBitrate.textContent = `${this.elements.bitrateSlider.value} kbps`;

        this._showNotification('Video loaded! Click Play to start.', 'success');
    }

    /**
     * Video metadata loaded
     */
    _onVideoMetadata() {
        const video = this.elements.originalVideo;
        this.elements.originalInfo.textContent = `${video.videoWidth}x${video.videoHeight}`;

        if (isFinite(video.duration)) {
            this.elements.duration.textContent = this._formatTime(video.duration);
            this.elements.seekBar.max = video.duration;
        }
    }

    /**
     * Start compression loopback
     */
    async _startCompression() {
        if (!this.elements.enableCompression.checked) {
            return;
        }

        // Get the media stream from video element
        let stream = this.mediaStream;

        // If it's a file video, capture it from the video element
        if (!stream && this.elements.originalVideo.src) {
            // Create a canvas to capture the video
            const canvas = document.createElement('canvas');
            const video = this.elements.originalVideo;
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
            const ctx = canvas.getContext('2d');

            // Create stream from canvas
            this.canvasStream = canvas.captureStream(30);
            stream = this.canvasStream;

            // Start drawing video to canvas
            const drawFrame = () => {
                if (video.paused || video.ended) return;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                requestAnimationFrame(drawFrame);
            };

            // Start when video plays
            if (!video.paused) {
                drawFrame();
            }
            video.addEventListener('play', drawFrame);
        }

        if (!stream) {
            console.warn('No stream available for compression');
            return;
        }

        // Configure and start compression
        this.compressionLoopback.updateConfig({
            codec: this.elements.codecSelect.value,
            bitrate: parseInt(this.elements.bitrateSlider.value),
            framerate: 30
        });

        await this.compressionLoopback.start(stream);
    }

    /**
     * Toggle play/pause
     */
    async _togglePlayPause() {
        if (!this.videoLoaded) return;

        try {
            if (this.elements.originalVideo.paused) {
                await this.elements.originalVideo.play();
                await this._startCompression();
            } else {
                this.elements.originalVideo.pause();
            }
        } catch (error) {
            this._showNotification(error.message, 'error');
        }
    }

    /**
     * Stop playback
     */
    _stop() {
        this.elements.originalVideo.pause();
        this.elements.originalVideo.currentTime = 0;
        this.compressionLoopback.stop();
    }

    /**
     * Play event handler
     */
    _onPlay() {
        this.elements.playPauseBtn.textContent = 'Pause';
    }

    /**
     * Pause event handler
     */
    _onPause() {
        this.elements.playPauseBtn.textContent = 'Play';
    }

    /**
     * Video ended handler
     */
    _onEnded() {
        this.elements.playPauseBtn.textContent = 'Play';
        this.compressionLoopback.stop();
    }

    /**
     * Seek to position
     */
    _seek(value) {
        if (isFinite(this.elements.originalVideo.duration)) {
            this.elements.originalVideo.currentTime = parseFloat(value);
        }
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
     * Update compression settings
     */
    async _updateCompressionSettings() {
        const enabled = this.elements.enableCompression.checked;
        const codec = this.elements.codecSelect.value;
        const bitrate = parseInt(this.elements.bitrateSlider.value);

        // Update display
        this.elements.statTargetBitrate.textContent = `${bitrate} kbps`;

        // Update compressed panel visibility
        document.querySelector('.compressed-panel').style.opacity = enabled ? '1' : '0.5';

        if (!enabled) {
            this.compressionLoopback.stop();
            this.elements.compressedVideo.srcObject = null;
            return;
        }

        // Restart compression with new settings if video is playing
        if (!this.elements.originalVideo.paused) {
            await this.compressionLoopback.stop();
            this.compressionLoopback.updateConfig({ codec, bitrate });
            await this._startCompression();
        }
    }

    /**
     * Update compression stats display
     */
    _updateCompressionStats(stats) {
        this.elements.statCodec.textContent = stats.codecName || this.elements.codecSelect.value;
        this.elements.statActualBitrate.textContent = `${stats.actualBitrate} kbps`;
        this.elements.statResolution.textContent = stats.frameWidth > 0 ? `${stats.frameWidth}x${stats.frameHeight}` : '-';
        this.elements.statFps.textContent = stats.framesPerSecond > 0 ? `${Math.round(stats.framesPerSecond)}` : '-';
        this.elements.statPacketsLost.textContent = stats.packetsLost.toString();
        this.elements.statBytesSent.textContent = this._formatBytes(stats.bytesSent);
        this.elements.statFramesDropped.textContent = stats.framesDropped.toString();

        // Update compressed info
        if (stats.frameWidth > 0) {
            this.elements.compressedInfo.textContent = `${stats.frameWidth}x${stats.frameHeight} @ ${stats.actualBitrate}kbps`;
        }
    }

    /**
     * Format bytes to human readable
     */
    _formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
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
                this._seek(this.elements.originalVideo.currentTime - 5);
                break;
            case 'ArrowRight':
                e.preventDefault();
                this._seek(this.elements.originalVideo.currentTime + 5);
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
    _cleanup() {
        this.compressionLoopback.stop();

        // Stop media stream tracks
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.canvasStream) {
            this.canvasStream.getTracks().forEach(track => track.stop());
            this.canvasStream = null;
        }

        // Clear video sources
        this.elements.originalVideo.srcObject = null;
        this.elements.originalVideo.src = '';
        this.elements.compressedVideo.srcObject = null;
    }

    /**
     * Dispose all resources
     */
    dispose() {
        this._cleanup();
        this.compressionLoopback.dispose();
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
