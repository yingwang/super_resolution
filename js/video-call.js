/**
 * Video Call Enhancement POC
 * Simulates a video call with receiver-side enhancement
 */

class VideoCallApp {
    constructor() {
        // DOM Elements
        this.elements = {
            // Videos
            localVideo: document.getElementById('localVideo'),
            enhancedCanvas: document.getElementById('enhancedVideo'),
            remoteVideoOriginal: document.getElementById('remoteVideoOriginal'),
            remoteOverlay: document.getElementById('remoteOverlay'),
            originalPanel: document.getElementById('originalPanel'),

            // Status
            connectionDot: document.getElementById('connectionDot'),
            connectionStatus: document.getElementById('connectionStatus'),
            latency: document.getElementById('latency'),
            enhancementStatus: document.getElementById('enhancementStatus'),

            // Controls
            startCallBtn: document.getElementById('startCallBtn'),
            endCallBtn: document.getElementById('endCallBtn'),
            toggleEnhanceBtn: document.getElementById('toggleEnhanceBtn'),
            toggleCompareBtn: document.getElementById('toggleCompareBtn'),

            // Settings
            enhancementType: document.getElementById('enhancementType'),
            intensity: document.getElementById('intensity'),
            intensityVal: document.getElementById('intensityVal'),
            brightness: document.getElementById('brightness'),
            brightnessVal: document.getElementById('brightnessVal'),
            contrast: document.getElementById('contrast'),
            contrastVal: document.getElementById('contrastVal'),

            // Simulation
            videoSource: document.getElementById('videoSource'),
            simulateQuality: document.getElementById('simulateQuality'),

            // Info
            remoteResolution: document.getElementById('remoteResolution'),
            enhancementMode: document.getElementById('enhancementMode'),

            // Stats
            fps: document.getElementById('fps'),
            frameTime: document.getElementById('frameTime'),
            inputRes: document.getElementById('inputRes'),
            outputRes: document.getElementById('outputRes')
        };

        // State
        this.isCallActive = false;
        this.isEnhancementEnabled = true;
        this.isCompareMode = false;
        this.localStream = null;
        this.remoteStream = null;

        // Enhancer
        this.enhancer = new VideoEnhancer();

        // Processing
        this.animationFrameId = null;
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

        // Quality simulation settings
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
            // Initialize the enhancer
            const backend = await this.enhancer.init(this.elements.enhancedCanvas);
            console.log('Enhancer initialized:', backend);

            this._updateEnhancementSettings();
            this._showNotification('Ready to start a call', 'success');

        } catch (error) {
            console.error('Init error:', error);
            this._showNotification(`Initialization failed: ${error.message}`, 'error');
        }
    }

    /**
     * Bind event listeners
     */
    _bindEvents() {
        // Call controls
        this.elements.startCallBtn.addEventListener('click', () => this.startCall());
        this.elements.endCallBtn.addEventListener('click', () => this.endCall());
        this.elements.toggleEnhanceBtn.addEventListener('click', () => this.toggleEnhancement());
        this.elements.toggleCompareBtn.addEventListener('click', () => this.toggleCompare());

        // Enhancement settings
        this.elements.enhancementType.addEventListener('change', () => this._updateEnhancementSettings());

        const sliders = [
            { el: this.elements.intensity, display: this.elements.intensityVal },
            { el: this.elements.brightness, display: this.elements.brightnessVal },
            { el: this.elements.contrast, display: this.elements.contrastVal }
        ];

        sliders.forEach(({ el, display }) => {
            el.addEventListener('input', () => {
                display.textContent = el.value;
                this._updateEnhancementSettings();
            });
        });

        // Simulation settings
        this.elements.simulateQuality.addEventListener('change', () => {
            if (this.isCallActive) {
                this._showNotification('Quality will change on next call', 'info');
            }
        });
    }

    /**
     * Start the video call
     */
    async startCall() {
        try {
            this._showNotification('Starting call...', 'info');
            this._updateConnectionStatus('connecting');

            const source = this.elements.videoSource.value;

            if (source === 'webcam') {
                // Use webcam as both local and "remote" (simulated)
                await this._startWebcamCall();
            } else {
                // Use sample video as "remote" video
                await this._startSampleCall(source);
            }

            this.isCallActive = true;
            this.elements.startCallBtn.disabled = true;
            this.elements.endCallBtn.disabled = false;
            this.elements.remoteOverlay.classList.add('hidden');
            this._updateConnectionStatus('connected');

            // Start processing loop
            this._startProcessing();

            this._showNotification('Call connected!', 'success');

        } catch (error) {
            console.error('Start call error:', error);
            this._updateConnectionStatus('disconnected');
            this._showNotification(`Failed to start call: ${error.message}`, 'error');
        }
    }

    /**
     * Start call using webcam
     */
    async _startWebcamCall() {
        const quality = this.qualitySettings[this.elements.simulateQuality.value];

        // Get webcam stream
        this.localStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: quality.width },
                height: { ideal: quality.height },
                facingMode: 'user'
            },
            audio: false
        });

        // Show local preview
        this.elements.localVideo.srcObject = this.localStream;

        // Simulate remote stream (in real app, this would come from WebRTC)
        // For POC, we use the same stream as "remote"
        this.remoteStream = this.localStream;
        this.elements.remoteVideoOriginal.srcObject = this.remoteStream;

        await this.elements.remoteVideoOriginal.play();
    }

    /**
     * Start call using sample video
     */
    async _startSampleCall(source) {
        const videoUrl = this.sampleVideos[source];
        const quality = this.qualitySettings[this.elements.simulateQuality.value];

        // Create a hidden video element for the sample
        this.elements.remoteVideoOriginal.src = videoUrl;
        this.elements.remoteVideoOriginal.crossOrigin = 'anonymous';
        this.elements.remoteVideoOriginal.loop = true;

        // Load and play
        await new Promise((resolve, reject) => {
            this.elements.remoteVideoOriginal.onloadeddata = resolve;
            this.elements.remoteVideoOriginal.onerror = reject;
            this.elements.remoteVideoOriginal.load();
        });

        await this.elements.remoteVideoOriginal.play();

        // Show placeholder for local video
        this.elements.localVideo.poster = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%23334155" width="100" height="100"/><text x="50" y="55" font-family="sans-serif" font-size="10" fill="%2394a3b8" text-anchor="middle">No Camera</text></svg>';

        this._updateResolutionInfo(quality.width, quality.height);
    }

    /**
     * End the video call
     */
    endCall() {
        // Stop processing
        this._stopProcessing();

        // Stop streams
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }

        // Reset video elements
        this.elements.localVideo.srcObject = null;
        this.elements.remoteVideoOriginal.srcObject = null;
        this.elements.remoteVideoOriginal.src = '';

        // Clear canvas
        const ctx = this.elements.enhancedCanvas.getContext('2d');
        ctx.clearRect(0, 0, this.elements.enhancedCanvas.width, this.elements.enhancedCanvas.height);

        // Update UI
        this.isCallActive = false;
        this.elements.startCallBtn.disabled = false;
        this.elements.endCallBtn.disabled = true;
        this.elements.remoteOverlay.classList.remove('hidden');
        this.elements.remoteOverlay.querySelector('span').textContent = 'Call ended';

        this._updateConnectionStatus('disconnected');
        this._resetStats();

        this._showNotification('Call ended', 'info');
    }

    /**
     * Toggle enhancement on/off
     */
    toggleEnhancement() {
        this.isEnhancementEnabled = !this.isEnhancementEnabled;

        this.elements.toggleEnhanceBtn.classList.toggle('active', this.isEnhancementEnabled);
        this.elements.toggleEnhanceBtn.innerHTML = this.isEnhancementEnabled
            ? '<span class="icon">✨</span> Enhancement On'
            : '<span class="icon">✨</span> Enhancement Off';

        this.elements.enhancementStatus.textContent = this.isEnhancementEnabled ? 'On' : 'Off';

        this._updateEnhancementSettings();
    }

    /**
     * Toggle comparison mode
     */
    toggleCompare() {
        this.isCompareMode = !this.isCompareMode;

        this.elements.toggleCompareBtn.classList.toggle('active', this.isCompareMode);
        this.elements.originalPanel.style.display = this.isCompareMode ? 'block' : 'none';
        document.querySelector('.main-content').classList.toggle('comparison-mode', this.isCompareMode);
    }

    /**
     * Update enhancement settings
     */
    _updateEnhancementSettings() {
        const mode = this.isEnhancementEnabled ? this.elements.enhancementType.value : 'passthrough';

        // Map 'auto' to 'enhance' for the enhancer
        const enhancerMode = mode === 'auto' ? 'enhance' : mode;

        this.enhancer.updateSettings({
            mode: enhancerMode,
            intensity: this.elements.intensity.value / 100,
            brightness: this.elements.brightness.value / 100,
            contrast: this.elements.contrast.value / 100,
            saturation: 0
        });

        // Update display
        const modeText = this.elements.enhancementType.options[this.elements.enhancementType.selectedIndex].text;
        this.elements.enhancementMode.textContent = this.isEnhancementEnabled ? modeText : 'Off';
        this.elements.enhancementStatus.textContent = this.isEnhancementEnabled ? 'On' : 'Off';
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
        if (!this.isCallActive) return;

        const startTime = performance.now();

        try {
            const video = this.elements.remoteVideoOriginal;

            if (video.readyState >= 2 && video.videoWidth > 0) {
                const inputWidth = video.videoWidth;
                const inputHeight = video.videoHeight;

                // Calculate output dimensions
                let outputWidth = inputWidth;
                let outputHeight = inputHeight;

                // For super resolution, double the output size
                if (this.isEnhancementEnabled && this.elements.enhancementType.value === 'superres') {
                    outputWidth = Math.min(inputWidth * 2, 1920);
                    outputHeight = Math.min(inputHeight * 2, 1080);
                }

                // Process frame through enhancer
                await this.enhancer.processFrame(video, outputWidth, outputHeight);

                // Update resolution info
                this._updateResolutionInfo(inputWidth, inputHeight, outputWidth, outputHeight);
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

            this._updateStatsDisplay();
        }

        // Schedule next frame
        this.animationFrameId = requestAnimationFrame(() => this._processFrame());
    }

    /**
     * Update connection status display
     */
    _updateConnectionStatus(status) {
        const dot = this.elements.connectionDot;
        const text = this.elements.connectionStatus;

        dot.classList.remove('connected');

        switch (status) {
            case 'connected':
                dot.classList.add('connected');
                text.textContent = 'Connected';
                this.elements.latency.textContent = '~50ms';
                break;
            case 'connecting':
                text.textContent = 'Connecting...';
                this.elements.latency.textContent = '--';
                break;
            default:
                text.textContent = 'Disconnected';
                this.elements.latency.textContent = '--';
        }
    }

    /**
     * Update resolution info display
     */
    _updateResolutionInfo(inputW, inputH, outputW = null, outputH = null) {
        this.elements.inputRes.textContent = `${inputW}x${inputH}`;
        this.elements.remoteResolution.textContent = `${inputW}x${inputH}`;

        if (outputW && outputH) {
            this.elements.outputRes.textContent = `${outputW}x${outputH}`;
        } else {
            this.elements.outputRes.textContent = `${inputW}x${inputH}`;
        }
    }

    /**
     * Update stats display
     */
    _updateStatsDisplay() {
        this.elements.fps.textContent = `${this.stats.fps} fps`;
        this.elements.frameTime.textContent = `${this.stats.frameTime} ms`;
    }

    /**
     * Reset stats display
     */
    _resetStats() {
        this.elements.fps.textContent = '--';
        this.elements.frameTime.textContent = '--';
        this.elements.inputRes.textContent = '--';
        this.elements.outputRes.textContent = '--';
        this.elements.remoteResolution.textContent = '--';
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
        this.endCall();
        this.enhancer.dispose();
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new VideoCallApp();
    window.app.init();
});

window.addEventListener('beforeunload', () => {
    if (window.app) {
        window.app.dispose();
    }
});
