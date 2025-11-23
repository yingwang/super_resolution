/**
 * Video Processor Module
 * Handles video stream acquisition via WebRTC and MediaStream APIs
 */

class VideoProcessor {
    constructor() {
        this.videoElement = null;
        this.outputCanvas = null;
        this.enhancer = null;
        this.mediaStream = null;

        this.isPlaying = false;
        this.isProcessing = false;
        this.animationFrameId = null;

        // Performance tracking
        this.stats = {
            fps: 0,
            frameTime: 0,
            frameCount: 0,
            lastTime: performance.now()
        };

        // Callbacks
        this.onStats = null;
        this.onError = null;
    }

    /**
     * Initialize the video processor
     */
    async init(videoElement, outputCanvas, enhancer) {
        this.videoElement = videoElement;
        this.outputCanvas = outputCanvas;
        this.enhancer = enhancer;

        // Configure video element
        this.videoElement.setAttribute('playsinline', '');
        this.videoElement.setAttribute('muted', '');
        this.videoElement.muted = true;

        // Add event listeners
        this.videoElement.addEventListener('loadedmetadata', () => this._onVideoLoaded());
        this.videoElement.addEventListener('play', () => this._startProcessing());
        this.videoElement.addEventListener('pause', () => this._stopProcessing());
        this.videoElement.addEventListener('ended', () => this._onVideoEnded());
        this.videoElement.addEventListener('error', (e) => this._onError(e));

        console.log('VideoProcessor initialized');
    }

    /**
     * Load video from URL
     */
    async loadFromUrl(url) {
        this._cleanup();

        return new Promise((resolve, reject) => {
            this.videoElement.src = url;
            this.videoElement.crossOrigin = 'anonymous';

            this.videoElement.onloadeddata = () => {
                console.log('Video loaded:', url);
                resolve({
                    width: this.videoElement.videoWidth,
                    height: this.videoElement.videoHeight,
                    duration: this.videoElement.duration
                });
            };

            this.videoElement.onerror = (e) => {
                reject(new Error(`Failed to load video: ${e.message || 'Unknown error'}`));
            };

            this.videoElement.load();
        });
    }

    /**
     * Load video from webcam using WebRTC
     */
    async loadFromWebcam(constraints = {}) {
        this._cleanup();

        const defaultConstraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: false
        };

        const mergedConstraints = {
            ...defaultConstraints,
            video: { ...defaultConstraints.video, ...constraints.video }
        };

        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia(mergedConstraints);
            this.videoElement.srcObject = this.mediaStream;

            return new Promise((resolve, reject) => {
                this.videoElement.onloadedmetadata = () => {
                    const track = this.mediaStream.getVideoTracks()[0];
                    const settings = track.getSettings();

                    console.log('Webcam stream acquired:', settings);
                    resolve({
                        width: settings.width,
                        height: settings.height,
                        duration: Infinity,
                        isLive: true
                    });
                };

                this.videoElement.onerror = reject;
            });
        } catch (error) {
            if (error.name === 'NotAllowedError') {
                throw new Error('Camera access denied. Please allow camera access and try again.');
            } else if (error.name === 'NotFoundError') {
                throw new Error('No camera found. Please connect a camera and try again.');
            }
            throw error;
        }
    }

    /**
     * Load video from screen capture
     */
    async loadFromScreen() {
        this._cleanup();

        try {
            this.mediaStream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    cursor: 'always',
                    displaySurface: 'monitor'
                },
                audio: false
            });

            this.videoElement.srcObject = this.mediaStream;

            // Handle stream ended (user stopped sharing)
            this.mediaStream.getVideoTracks()[0].addEventListener('ended', () => {
                this._onVideoEnded();
            });

            return new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => {
                    const track = this.mediaStream.getVideoTracks()[0];
                    const settings = track.getSettings();

                    resolve({
                        width: settings.width,
                        height: settings.height,
                        duration: Infinity,
                        isLive: true
                    });
                };
            });
        } catch (error) {
            if (error.name === 'NotAllowedError') {
                throw new Error('Screen sharing was cancelled.');
            }
            throw error;
        }
    }

    /**
     * Create a WebRTC peer connection for streaming
     * This is useful for receiving video from a remote peer
     */
    async createPeerConnection(configuration = {}) {
        const defaultConfig = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };

        const pc = new RTCPeerConnection({ ...defaultConfig, ...configuration });

        pc.ontrack = (event) => {
            if (event.streams && event.streams[0]) {
                this.mediaStream = event.streams[0];
                this.videoElement.srcObject = this.mediaStream;
            }
        };

        pc.oniceconnectionstatechange = () => {
            console.log('ICE connection state:', pc.iceConnectionState);
        };

        return pc;
    }

    /**
     * Play video
     */
    async play() {
        if (!this.videoElement.src && !this.videoElement.srcObject) {
            throw new Error('No video source loaded');
        }

        try {
            await this.videoElement.play();
            this.isPlaying = true;
        } catch (error) {
            if (error.name === 'NotAllowedError') {
                // Autoplay was prevented, user interaction required
                console.warn('Autoplay prevented, user interaction required');
            }
            throw error;
        }
    }

    /**
     * Pause video
     */
    pause() {
        this.videoElement.pause();
        this.isPlaying = false;
    }

    /**
     * Stop video
     */
    stop() {
        this.pause();
        this.videoElement.currentTime = 0;
        this._stopProcessing();
    }

    /**
     * Seek to position
     */
    seek(time) {
        if (isFinite(this.videoElement.duration)) {
            this.videoElement.currentTime = time;
        }
    }

    /**
     * Get current playback info
     */
    getPlaybackInfo() {
        return {
            currentTime: this.videoElement.currentTime,
            duration: this.videoElement.duration,
            paused: this.videoElement.paused,
            ended: this.videoElement.ended,
            width: this.videoElement.videoWidth,
            height: this.videoElement.videoHeight
        };
    }

    /**
     * Video loaded callback
     */
    _onVideoLoaded() {
        console.log('Video metadata loaded:', {
            width: this.videoElement.videoWidth,
            height: this.videoElement.videoHeight,
            duration: this.videoElement.duration
        });
    }

    /**
     * Start frame processing loop
     */
    _startProcessing() {
        if (this.isProcessing) return;

        this.isProcessing = true;
        this.stats.lastTime = performance.now();
        this.stats.frameCount = 0;

        this._processFrame();
    }

    /**
     * Stop frame processing
     */
    _stopProcessing() {
        this.isProcessing = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Process a single frame
     */
    async _processFrame() {
        if (!this.isProcessing) return;

        const startTime = performance.now();

        try {
            // Check if video has valid dimensions
            if (this.videoElement.videoWidth > 0 && this.videoElement.videoHeight > 0) {
                // Determine output dimensions
                const outputWidth = this.outputCanvas.clientWidth || this.videoElement.videoWidth;
                const outputHeight = this.outputCanvas.clientHeight || this.videoElement.videoHeight;

                // Process frame through enhancer
                await this.enhancer.processFrame(
                    this.videoElement,
                    outputWidth,
                    outputHeight
                );
            }
        } catch (error) {
            console.error('Frame processing error:', error);
        }

        // Update performance stats
        const frameTime = performance.now() - startTime;
        this.stats.frameCount++;

        const elapsed = performance.now() - this.stats.lastTime;
        if (elapsed >= 1000) {
            this.stats.fps = Math.round((this.stats.frameCount * 1000) / elapsed);
            this.stats.frameTime = Math.round(frameTime * 10) / 10;
            this.stats.frameCount = 0;
            this.stats.lastTime = performance.now();

            if (this.onStats) {
                this.onStats(this.stats);
            }
        }

        // Schedule next frame
        this.animationFrameId = requestAnimationFrame(() => this._processFrame());
    }

    /**
     * Video ended callback
     */
    _onVideoEnded() {
        this.isPlaying = false;
        this._stopProcessing();
        console.log('Video playback ended');
    }

    /**
     * Error callback
     */
    _onError(error) {
        console.error('Video error:', error);
        if (this.onError) {
            this.onError(error);
        }
    }

    /**
     * Cleanup resources
     */
    _cleanup() {
        this._stopProcessing();

        // Stop media stream tracks
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Clear video source
        this.videoElement.srcObject = null;
        this.videoElement.src = '';
    }

    /**
     * Dispose all resources
     */
    dispose() {
        this._cleanup();
        this.videoElement = null;
        this.outputCanvas = null;
        this.enhancer = null;
    }
}

// Export for use in other modules
window.VideoProcessor = VideoProcessor;
