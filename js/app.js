/**
 * Video Compression & Enhancement Comparison App
 * Three-way comparison: Original vs Received (Compressed) vs Enhanced
 */

class App {
    constructor() {
        // DOM elements
        this.elements = {
            // Video elements
            originalVideo: document.getElementById('originalVideo'),
            receivedVideo: document.getElementById('receivedVideo'),
            enhancedCanvas: document.getElementById('enhancedCanvas'),

            // Source controls
            sourceRadios: document.querySelectorAll('input[name="source"]'),
            urlInput: document.getElementById('urlInput'),
            videoUrlField: document.getElementById('videoUrl'),
            loadUrlBtn: document.getElementById('loadUrl'),
            sampleVideos: document.getElementById('sampleVideos'),
            sampleSelect: document.getElementById('sampleSelect'),
            loadSampleBtn: document.getElementById('loadSample'),

            // Compression controls
            codecSelect: document.getElementById('codecSelect'),
            bitrateSelect: document.getElementById('bitrateSelect'),
            bitrateSlider: document.getElementById('bitrateSlider'),
            bitrateValue: document.getElementById('bitrateValue'),

            // Enhancement controls
            enhancementMode: document.getElementById('enhancementMode'),
            intensity: document.getElementById('intensity'),
            intensityValue: document.getElementById('intensityValue'),
            brightness: document.getElementById('brightness'),
            brightnessValue: document.getElementById('brightnessValue'),
            contrast: document.getElementById('contrast'),
            contrastValue: document.getElementById('contrastValue'),

            // Playback controls
            playPauseBtn: document.getElementById('playPause'),
            stopBtn: document.getElementById('stop'),
            fullscreenBtn: document.getElementById('fullscreen'),
            currentTime: document.getElementById('currentTime'),
            duration: document.getElementById('duration'),
            seekBar: document.getElementById('seekBar'),

            // Info displays
            originalInfo: document.getElementById('originalInfo'),
            receivedInfo: document.getElementById('receivedInfo'),
            enhancedInfo: document.getElementById('enhancedInfo'),

            // Stats
            statCodec: document.getElementById('statCodec'),
            statTargetBitrate: document.getElementById('statTargetBitrate'),
            statActualBitrate: document.getElementById('statActualBitrate'),
            statResolution: document.getElementById('statResolution'),
            statEnhancement: document.getElementById('statEnhancement'),
            statFps: document.getElementById('statFps'),

            // Segmentation controls
            enableSegmentation: document.getElementById('enableSegmentation'),
            backgroundType: document.getElementById('backgroundType'),
            blurAmount: document.getElementById('blurAmount'),
            blurAmountValue: document.getElementById('blurAmountValue'),
            blurAmountGroup: document.getElementById('blurAmountGroup'),
            bgColor: document.getElementById('bgColor'),
            bgColorGroup: document.getElementById('bgColorGroup'),
            bgImageUrl: document.getElementById('bgImageUrl'),
            bgImageGroup: document.getElementById('bgImageGroup'),
            loadBgImage: document.getElementById('loadBgImage'),

            // Panel visibility controls
            showOriginal: document.getElementById('showOriginal'),
            showReceived: document.getElementById('showReceived'),
            showEnhanced: document.getElementById('showEnhanced'),
            originalPanel: document.getElementById('originalPanel'),
            receivedPanel: document.getElementById('receivedPanel'),
            enhancedPanel: document.getElementById('enhancedPanel')
        };

        // Modules
        this.compressionLoopback = new CompressionLoopback();
        this.enhancer = new VideoEnhancer();
        this.segmentation = new BackgroundSegmentation();
        this.espcn = new ESPCNSuperResolution();

        // State
        this.currentSource = 'sample';
        this.videoLoaded = false;
        this.mediaStream = null;
        this.canvasStream = null;
        this.isProcessing = false;
        this.animationFrameId = null;

        // Performance tracking
        this.stats = {
            frameCount: 0,
            lastTime: performance.now(),
            fps: 0
        };

        // Bind methods
        this._bindEvents();
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            this._showNotification('Initializing...', 'info');

            // Initialize enhancer
            await this.enhancer.init(this.elements.enhancedCanvas);
            console.log('Enhancer initialized');

            // Set up compression loopback callbacks
            this.compressionLoopback.onStats = (stats) => this._updateCompressionStats(stats);
            this.compressionLoopback.onCompressedStream = (stream) => {
                this.elements.receivedVideo.srcObject = stream;
            };

            // Configure video elements
            this.elements.originalVideo.setAttribute('playsinline', '');
            this.elements.originalVideo.setAttribute('muted', '');
            this.elements.originalVideo.muted = true;

            // Update initial enhancement settings
            this._updateEnhancementSettings();

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

        // Compression controls
        this.elements.codecSelect.addEventListener('change', () => this._updateCompressionSettings());

        // Bitrate controls
        this.elements.bitrateSelect.addEventListener('change', (e) => {
            this.elements.bitrateSlider.value = e.target.value;
            this.elements.bitrateValue.textContent = e.target.value;
            this._updateCompressionSettings();
        });

        this.elements.bitrateSlider.addEventListener('input', (e) => {
            this.elements.bitrateValue.textContent = e.target.value;
            this._updateCompressionSettings();
        });

        // Enhancement controls
        this.elements.enhancementMode.addEventListener('change', () => this._updateEnhancementSettings());

        const sliders = ['intensity', 'brightness', 'contrast'];
        sliders.forEach(name => {
            this.elements[name].addEventListener('input', (e) => {
                this.elements[`${name}Value`].textContent = e.target.value;
                this._updateEnhancementSettings();
            });
        });

        // Segmentation controls
        this.elements.enableSegmentation.addEventListener('change', () => this._updateSegmentationSettings());
        this.elements.backgroundType.addEventListener('change', () => this._updateSegmentationSettings());
        this.elements.blurAmount.addEventListener('input', (e) => {
            this.elements.blurAmountValue.textContent = e.target.value;
            this._updateSegmentationSettings();
        });
        this.elements.bgColor.addEventListener('input', () => this._updateSegmentationSettings());
        this.elements.loadBgImage.addEventListener('click', () => this._loadBackgroundImage());

        // Panel visibility controls
        this.elements.showOriginal.addEventListener('change', () => this._updatePanelVisibility());
        this.elements.showReceived.addEventListener('change', () => this._updatePanelVisibility());
        this.elements.showEnhanced.addEventListener('change', () => this._updatePanelVisibility());

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

            this.elements.originalVideo.onerror = () => {
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
            await this._startProcessing();

        } catch (error) {
            this._showNotification(error.message || 'Failed to access webcam', 'error');
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
            await this._startProcessing();

        } catch (error) {
            this._showNotification(error.message || 'Screen sharing was cancelled', 'error');
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

        // Update stats
        this.elements.statTargetBitrate.textContent = `${this.elements.bitrateSlider.value} kbps`;
        this.elements.statResolution.textContent = `${info.width}x${info.height}`;

        this._showNotification('Video loaded! Click Play to start.', 'success');
    }

    /**
     * Video metadata loaded
     */
    _onVideoMetadata() {
        const video = this.elements.originalVideo;
        this.elements.originalInfo.textContent = `${video.videoWidth}x${video.videoHeight}`;
        this.elements.statResolution.textContent = `${video.videoWidth}x${video.videoHeight}`;

        if (isFinite(video.duration)) {
            this.elements.duration.textContent = this._formatTime(video.duration);
            this.elements.seekBar.max = video.duration;
        }
    }

    /**
     * Start compression and enhancement processing
     */
    async _startProcessing() {
        // Get the media stream from video element
        let stream = this.mediaStream;

        // If it's a file video, capture it from the video element
        if (!stream && this.elements.originalVideo.src) {
            const canvas = document.createElement('canvas');
            const video = this.elements.originalVideo;
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
            const ctx = canvas.getContext('2d');

            this.canvasStream = canvas.captureStream(30);
            stream = this.canvasStream;

            // Draw video to canvas
            const drawFrame = () => {
                if (video.paused || video.ended) return;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                requestAnimationFrame(drawFrame);
            };

            if (!video.paused) {
                drawFrame();
            }
            video.addEventListener('play', drawFrame);
        }

        if (!stream) {
            console.warn('No stream available');
            return;
        }

        // Configure and start compression
        this.compressionLoopback.updateConfig({
            codec: this.elements.codecSelect.value,
            bitrate: parseInt(this.elements.bitrateSlider.value),
            framerate: 30
        });

        await this.compressionLoopback.start(stream);

        // Start enhancement processing loop
        this._startEnhancementLoop();
    }

    /**
     * Start the enhancement processing loop
     */
    _startEnhancementLoop() {
        if (this.isProcessing) return;

        this.isProcessing = true;
        this.stats.lastTime = performance.now();
        this.stats.frameCount = 0;

        // Create temp canvas for segmentation output
        const segCanvas = document.createElement('canvas');
        const segCtx = segCanvas.getContext('2d');

        const processFrame = async () => {
            if (!this.isProcessing) return;

            const receivedVideo = this.elements.receivedVideo;

            // Only process if received video has valid dimensions
            if (receivedVideo.videoWidth > 0 && receivedVideo.videoHeight > 0) {
                try {
                    const outputWidth = receivedVideo.videoWidth;
                    const outputHeight = receivedVideo.videoHeight;

                    // Resize segmentation canvas if needed
                    if (segCanvas.width !== outputWidth || segCanvas.height !== outputHeight) {
                        segCanvas.width = outputWidth;
                        segCanvas.height = outputHeight;
                    }

                    let sourceForEnhancer = receivedVideo;

                    // Apply background segmentation if enabled
                    if (this.segmentation.isEnabled && this.segmentation.isReady()) {
                        const segmented = await this.segmentation.processFrame(receivedVideo, segCanvas);
                        if (segmented) {
                            sourceForEnhancer = segCanvas;
                        }
                    }

                    // Process frame - use ESPCN or regular enhancer
                    const mode = this.elements.enhancementMode.value;
                    if (mode === 'espcn' && this.espcn.isReady) {
                        const intensity = this.elements.intensity.value / 100;
                        await this.espcn.processFrameHybrid(
                            sourceForEnhancer,
                            this.elements.enhancedCanvas,
                            intensity
                        );
                    } else {
                        await this.enhancer.processFrame(sourceForEnhancer, outputWidth, outputHeight);
                    }

                    // Update FPS stats
                    this.stats.frameCount++;
                    const now = performance.now();
                    const elapsed = now - this.stats.lastTime;

                    if (elapsed >= 1000) {
                        this.stats.fps = Math.round((this.stats.frameCount * 1000) / elapsed);
                        this.stats.frameCount = 0;
                        this.stats.lastTime = now;
                        this.elements.statFps.textContent = `${this.stats.fps}`;
                    }
                } catch (error) {
                    console.error('Enhancement error:', error);
                }
            }

            this.animationFrameId = requestAnimationFrame(processFrame);
        };

        processFrame();
    }

    /**
     * Stop enhancement processing
     */
    _stopEnhancementLoop() {
        this.isProcessing = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Toggle play/pause
     */
    async _togglePlayPause() {
        if (!this.videoLoaded) return;

        try {
            if (this.elements.originalVideo.paused) {
                await this.elements.originalVideo.play();
                await this._startProcessing();
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
        this._stopEnhancementLoop();
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
        this._stopEnhancementLoop();
    }

    /**
     * Video ended handler
     */
    _onEnded() {
        this.elements.playPauseBtn.textContent = 'Play';
        this._stopEnhancementLoop();
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
        const codec = this.elements.codecSelect.value;
        const bitrate = parseInt(this.elements.bitrateSlider.value);

        // Update display
        this.elements.statTargetBitrate.textContent = `${bitrate} kbps`;

        // Restart compression with new settings if video is playing
        if (!this.elements.originalVideo.paused) {
            await this.compressionLoopback.stop();
            this.compressionLoopback.updateConfig({ codec, bitrate });
            await this._startProcessing();
        }
    }

    /**
     * Update enhancement settings
     */
    async _updateEnhancementSettings() {
        const mode = this.elements.enhancementMode.value;
        const intensity = this.elements.intensity.value / 100;
        const brightness = this.elements.brightness.value / 100;
        const contrast = this.elements.contrast.value / 100;

        // Initialize ESPCN if selected and not ready
        if (mode === 'espcn' && !this.espcn.isReady) {
            this._showNotification('Loading ESPCN model...', 'info');
            try {
                await this.espcn.init();
                const info = this.espcn.getModelInfo();
                const weightType = info.pretrained ? '✓ Trained weights' : '⚠ Crafted weights';
                this._showNotification(`ESPCN ready! ${weightType} (${info.params} params)`, 'success');
                console.log('ESPCN Model Info:', info);
            } catch (error) {
                this._showNotification('Failed to load ESPCN model', 'error');
                this.elements.enhancementMode.value = 'sharpen';
                return;
            }
        }

        this.enhancer.updateSettings({
            mode,
            intensity,
            brightness,
            contrast,
            saturation: 0
        });

        // Update display
        const modeText = this.elements.enhancementMode.options[this.elements.enhancementMode.selectedIndex].text;
        this.elements.statEnhancement.textContent = modeText;
        this.elements.enhancedInfo.textContent = `${modeText} @ ${this.elements.intensity.value}%`;
    }

    /**
     * Update segmentation settings
     */
    async _updateSegmentationSettings() {
        const enabled = this.elements.enableSegmentation.checked;
        const backgroundType = this.elements.backgroundType.value;

        // Show/hide relevant controls
        this.elements.blurAmountGroup.style.display = backgroundType === 'blur' ? 'flex' : 'none';
        this.elements.bgColorGroup.style.display = (backgroundType === 'color' || backgroundType === 'image') ? 'flex' : 'none';
        this.elements.bgImageGroup.style.display = backgroundType === 'image' ? 'block' : 'none';

        // Initialize segmentation model if enabling
        if (enabled && !this.segmentation.isReady()) {
            this._showNotification('Loading BodyPix model...', 'info');
            try {
                await this.segmentation.init();
                this._showNotification('Background segmentation ready!', 'success');
            } catch (error) {
                this._showNotification('Failed to load segmentation model', 'error');
                this.elements.enableSegmentation.checked = false;
                return;
            }
        }

        // Update segmentation settings
        this.segmentation.setEnabled(enabled);
        this.segmentation.updateSettings({
            backgroundType,
            blurAmount: parseInt(this.elements.blurAmount.value),
            backgroundColor: this.elements.bgColor.value
        });
    }

    /**
     * Load background image for virtual background
     */
    async _loadBackgroundImage() {
        const url = this.elements.bgImageUrl.value.trim();
        if (!url) {
            this._showNotification('Please enter an image URL', 'error');
            return;
        }

        try {
            this._showNotification('Loading background image...', 'info');
            await this.segmentation.setBackgroundImage(url);
            this._showNotification('Background image loaded!', 'success');
        } catch (error) {
            this._showNotification('Failed to load background image', 'error');
        }
    }

    /**
     * Update panel visibility based on checkboxes
     */
    _updatePanelVisibility() {
        const showOriginal = this.elements.showOriginal.checked;
        const showReceived = this.elements.showReceived.checked;
        const showEnhanced = this.elements.showEnhanced.checked;

        this.elements.originalPanel.classList.toggle('hidden', !showOriginal);
        this.elements.receivedPanel.classList.toggle('hidden', !showReceived);
        this.elements.enhancedPanel.classList.toggle('hidden', !showEnhanced);
    }

    /**
     * Update compression stats display
     */
    _updateCompressionStats(stats) {
        this.elements.statCodec.textContent = stats.codecName || this.elements.codecSelect.value;
        this.elements.statActualBitrate.textContent = `${stats.actualBitrate} kbps`;

        // Update received info
        if (stats.frameWidth > 0) {
            this.elements.receivedInfo.textContent = `${stats.frameWidth}x${stats.frameHeight} @ ${stats.actualBitrate}kbps`;
        }
    }

    /**
     * Handle keyboard shortcuts
     */
    _handleKeyboard(e) {
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
        document.querySelectorAll('.notification').forEach(n => n.remove());

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    /**
     * Cleanup resources
     */
    _cleanup() {
        this._stopEnhancementLoop();
        this.compressionLoopback.stop();

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.canvasStream) {
            this.canvasStream.getTracks().forEach(track => track.stop());
            this.canvasStream = null;
        }

        this.elements.originalVideo.srcObject = null;
        this.elements.originalVideo.src = '';
        this.elements.receivedVideo.srcObject = null;

        // Reset button state
        this.elements.playPauseBtn.textContent = 'Play';
    }

    /**
     * Dispose all resources
     */
    dispose() {
        this._cleanup();
        this.compressionLoopback.dispose();
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
