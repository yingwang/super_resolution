/**
 * Video Enhancement POC - Receiver Side
 * Simulates receiving a video stream and applying local enhancement
 */

class VideoEnhancementApp {
    constructor() {
        // DOM Elements
        this.elements = {
            // Videos
            originalVideo: document.getElementById('originalVideo'),
            enhancedCanvas: document.getElementById('enhancedCanvas'),
            originalOverlay: document.getElementById('originalOverlay'),
            enhancedOverlay: document.getElementById('enhancedOverlay'),

            // Source controls
            videoSource: document.getElementById('videoSource'),
            simulateQuality: document.getElementById('simulateQuality'),
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),

            // Enhancement controls
            enableEnhancement: document.getElementById('enableEnhancement'),
            enhancementMode: document.getElementById('enhancementMode'),
            intensity: document.getElementById('intensity'),
            intensityVal: document.getElementById('intensityVal'),
            brightness: document.getElementById('brightness'),
            brightnessVal: document.getElementById('brightnessVal'),
            contrast: document.getElementById('contrast'),
            contrastVal: document.getElementById('contrastVal'),
            saturation: document.getElementById('saturation'),
            saturationVal: document.getElementById('saturationVal'),

            // Resolution badges
            originalRes: document.getElementById('originalRes'),
            enhancedRes: document.getElementById('enhancedRes'),

            // Stats
            fps: document.getElementById('fps'),
            frameTime: document.getElementById('frameTime'),
            inputRes: document.getElementById('inputRes'),
            outputRes: document.getElementById('outputRes'),
            backend: document.getElementById('backend')
        };

        // State
        this.isRunning = false;
        this.mediaStream = null;
        this.animationFrameId = null;

        // Enhancer
        this.enhancer = new VideoEnhancer();

        // Stats
        this.stats = {
            frameCount: 0,
            lastTime: performance.now(),
            fps: 0,
            frameTime: 0
        };

        // Sample video URLs
        this.sampleVideos = {
            sample1: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
            sample2: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
            sample3: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4'
        };

        // Quality settings (simulated degradation)
        this.qualitySettings = {
            good: { width: 1280, height: 720 },
            medium: { width: 854, height: 480 },
            poor: { width: 640, height: 360 },
            bad: { width: 426, height: 240 }
        };

        this._bindEvents();
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            // Initialize enhancer
            const backend = await this.enhancer.init(this.elements.enhancedCanvas);
            console.log('Enhancer initialized:', backend);

            // Show backend info
            const backendInfo = this.enhancer.getBackendInfo();
            this.elements.backend.textContent = backendInfo.tensorflow;

            this._updateEnhancementSettings();
            this._showNotification('Ready! Select a source and click Start.', 'success');

        } catch (error) {
            console.error('Init error:', error);
            this._showNotification(`Initialization failed: ${error.message}`, 'error');
        }
    }

    /**
     * Bind event listeners
     */
    _bindEvents() {
        // Start/Stop buttons
        this.elements.startBtn.addEventListener('click', () => this.start());
        this.elements.stopBtn.addEventListener('click', () => this.stop());

        // Enhancement toggle
        this.elements.enableEnhancement.addEventListener('change', () => this._updateEnhancementSettings());
        this.elements.enhancementMode.addEventListener('change', () => this._updateEnhancementSettings());

        // Sliders
        const sliders = [
            { el: this.elements.intensity, display: this.elements.intensityVal },
            { el: this.elements.brightness, display: this.elements.brightnessVal },
            { el: this.elements.contrast, display: this.elements.contrastVal },
            { el: this.elements.saturation, display: this.elements.saturationVal }
        ];

        sliders.forEach(({ el, display }) => {
            el.addEventListener('input', () => {
                display.textContent = el.value;
                this._updateEnhancementSettings();
            });
        });
    }

    /**
     * Start video processing
     */
    async start() {
        try {
            this._showNotification('Loading video...', 'info');

            const source = this.elements.videoSource.value;
            const quality = this.qualitySettings[this.elements.simulateQuality.value];

            if (source === 'webcam') {
                await this._loadWebcam(quality);
            } else {
                await this._loadSampleVideo(source);
            }

            // Update UI
            this.isRunning = true;
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.elements.originalOverlay.classList.add('hidden');
            this.elements.enhancedOverlay.classList.add('hidden');

            // Start processing
            this._startProcessing();

            this._showNotification('Video started!', 'success');

        } catch (error) {
            console.error('Start error:', error);
            this._showNotification(`Failed to start: ${error.message}`, 'error');
        }
    }

    /**
     * Load webcam stream
     */
    async _loadWebcam(quality) {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: quality.width },
                height: { ideal: quality.height },
                facingMode: 'user'
            },
            audio: false
        });

        this.elements.originalVideo.srcObject = this.mediaStream;
        await this.elements.originalVideo.play();
    }

    /**
     * Load sample video
     */
    async _loadSampleVideo(source) {
        const url = this.sampleVideos[source];

        this.elements.originalVideo.src = url;
        this.elements.originalVideo.crossOrigin = 'anonymous';
        this.elements.originalVideo.loop = true;

        await new Promise((resolve, reject) => {
            this.elements.originalVideo.onloadeddata = resolve;
            this.elements.originalVideo.onerror = () => reject(new Error('Failed to load video'));
            this.elements.originalVideo.load();
        });

        await this.elements.originalVideo.play();
    }

    /**
     * Stop video processing
     */
    stop() {
        // Stop processing loop
        this._stopProcessing();

        // Stop webcam if active
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Reset video
        this.elements.originalVideo.pause();
        this.elements.originalVideo.srcObject = null;
        this.elements.originalVideo.src = '';

        // Clear canvas
        const ctx = this.elements.enhancedCanvas.getContext('2d');
        ctx.clearRect(0, 0, this.elements.enhancedCanvas.width, this.elements.enhancedCanvas.height);

        // Update UI
        this.isRunning = false;
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this.elements.originalOverlay.classList.remove('hidden');
        this.elements.enhancedOverlay.classList.remove('hidden');
        this.elements.originalOverlay.querySelector('span').textContent = 'Stopped';
        this.elements.enhancedOverlay.querySelector('span').textContent = 'Stopped';

        // Reset stats
        this._resetStats();

        this._showNotification('Video stopped', 'info');
    }

    /**
     * Start frame processing loop
     */
    _startProcessing() {
        if (this.animationFrameId) return;

        this.stats.lastTime = performance.now();
        this.stats.frameCount = 0;

        this._processFrame();
    }

    /**
     * Stop frame processing
     */
    _stopProcessing() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Process a single frame
     */
    async _processFrame() {
        if (!this.isRunning) return;

        const startTime = performance.now();

        try {
            const video = this.elements.originalVideo;

            if (video.readyState >= 2 && video.videoWidth > 0) {
                const inputWidth = video.videoWidth;
                const inputHeight = video.videoHeight;

                // Calculate output dimensions
                let outputWidth = inputWidth;
                let outputHeight = inputHeight;

                // For super resolution, scale up
                const mode = this.elements.enhancementMode.value;
                if (this.elements.enableEnhancement.checked && mode === 'superres') {
                    outputWidth = Math.min(inputWidth * 2, 1920);
                    outputHeight = Math.min(inputHeight * 2, 1080);
                }

                // Process frame through enhancer
                await this.enhancer.processFrame(video, outputWidth, outputHeight);

                // Update resolution displays
                this.elements.originalRes.textContent = `${inputWidth}x${inputHeight}`;
                this.elements.enhancedRes.textContent = `${outputWidth}x${outputHeight}`;
                this.elements.inputRes.textContent = `${inputWidth}x${inputHeight}`;
                this.elements.outputRes.textContent = `${outputWidth}x${outputHeight}`;
            }
        } catch (error) {
            console.error('Frame processing error:', error);
        }

        // Update stats
        const frameTime = performance.now() - startTime;
        this.stats.frameCount++;

        const elapsed = performance.now() - this.stats.lastTime;
        if (elapsed >= 1000) {
            this.stats.fps = Math.round((this.stats.frameCount * 1000) / elapsed);
            this.stats.frameTime = Math.round(frameTime * 10) / 10;
            this.stats.frameCount = 0;
            this.stats.lastTime = performance.now();

            this.elements.fps.textContent = `${this.stats.fps}`;
            this.elements.frameTime.textContent = `${this.stats.frameTime}ms`;
        }

        // Schedule next frame
        this.animationFrameId = requestAnimationFrame(() => this._processFrame());
    }

    /**
     * Update enhancement settings
     */
    _updateEnhancementSettings() {
        const enabled = this.elements.enableEnhancement.checked;
        const mode = enabled ? this.elements.enhancementMode.value : 'passthrough';

        this.enhancer.updateSettings({
            mode: mode,
            intensity: this.elements.intensity.value / 100,
            brightness: this.elements.brightness.value / 100,
            contrast: this.elements.contrast.value / 100,
            saturation: this.elements.saturation.value / 100
        });
    }

    /**
     * Reset stats display
     */
    _resetStats() {
        this.elements.fps.textContent = '--';
        this.elements.frameTime.textContent = '--';
        this.elements.inputRes.textContent = '--';
        this.elements.outputRes.textContent = '--';
        this.elements.originalRes.textContent = '--';
        this.elements.enhancedRes.textContent = '--';
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
     * Cleanup
     */
    dispose() {
        this.stop();
        this.enhancer.dispose();
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new VideoEnhancementApp();
    window.app.init();
});

window.addEventListener('beforeunload', () => {
    if (window.app) {
        window.app.dispose();
    }
});
