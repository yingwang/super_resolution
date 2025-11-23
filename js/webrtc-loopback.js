/**
 * WebRTC Video Enhancement POC
 * Creates a real WebRTC loopback connection with actual video encoding/decoding
 */

class WebRTCEnhancementApp {
    constructor() {
        // DOM Elements
        this.elements = {
            // Videos
            sourceVideo: document.getElementById('sourceVideo'),
            receivedVideo: document.getElementById('receivedVideo'),
            enhancedCanvas: document.getElementById('enhancedCanvas'),
            originalOverlay: document.getElementById('originalOverlay'),
            enhancedOverlay: document.getElementById('enhancedOverlay'),

            // Status
            connectionDot: document.getElementById('connectionDot'),
            connectionStatus: document.getElementById('connectionStatus'),
            codecInfo: document.getElementById('codecInfo'),
            bitrateInfo: document.getElementById('bitrateInfo'),
            packetLoss: document.getElementById('packetLoss'),

            // Controls
            videoSource: document.getElementById('videoSource'),
            targetBitrate: document.getElementById('targetBitrate'),
            videoCodec: document.getElementById('videoCodec'),
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

        // WebRTC
        this.senderPC = null;
        this.receiverPC = null;
        this.mediaStream = null;
        this.statsInterval = null;
        this.lastBytesReceived = 0;
        this.lastStatsTime = 0;

        // State
        this.isRunning = false;
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

        // Sample video URLs (these support CORS)
        this.sampleVideos = {
            sample1: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
            sample2: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
            sample3: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4'
        };

        this._bindEvents();
    }

    /**
     * Initialize
     */
    async init() {
        try {
            const backend = await this.enhancer.init(this.elements.enhancedCanvas);
            console.log('Enhancer initialized:', backend);

            const backendInfo = this.enhancer.getBackendInfo();
            this.elements.backend.textContent = backendInfo.tensorflow;

            this._updateEnhancementSettings();
            this._showNotification('Ready! Select Webcam and click Start.', 'success');
        } catch (error) {
            console.error('Init error:', error);
            this._showNotification(`Init failed: ${error.message}`, 'error');
        }
    }

    /**
     * Bind events
     */
    _bindEvents() {
        this.elements.startBtn.addEventListener('click', () => this.start());
        this.elements.stopBtn.addEventListener('click', () => this.stop());

        // Bitrate change during call
        this.elements.targetBitrate.addEventListener('change', () => {
            if (this.isRunning) {
                this._updateBitrate();
            }
        });

        // Enhancement controls
        this.elements.enableEnhancement.addEventListener('change', () => this._updateEnhancementSettings());
        this.elements.enhancementMode.addEventListener('change', () => this._updateEnhancementSettings());

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
     * Start WebRTC loopback
     */
    async start() {
        try {
            this._showNotification('Setting up WebRTC...', 'info');
            this._updateConnectionStatus('connecting');

            const source = this.elements.videoSource.value;

            if (source === 'webcam') {
                // Use webcam directly
                await this._setupWebcam();
            } else {
                // For sample videos, use canvas capture approach
                await this._setupSampleVideo(source);
            }

            // Create WebRTC loopback
            await this._createWebRTCLoopback();

            // Update UI
            this.isRunning = true;
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.elements.originalOverlay.classList.add('hidden');
            this.elements.enhancedOverlay.classList.add('hidden');

            // Start processing
            this._startProcessing();

            // Start stats monitoring
            this._startStatsMonitoring();

            this._updateConnectionStatus('connected');
            this._showNotification('WebRTC connected!', 'success');

        } catch (error) {
            console.error('Start error:', error);
            this._updateConnectionStatus('disconnected');
            this._showNotification(`Failed: ${error.message}`, 'error');
            this.stop();
        }
    }

    /**
     * Setup webcam stream
     */
    async _setupWebcam() {
        console.log('Setting up webcam...');

        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            },
            audio: false
        });

        // Show source preview
        this.elements.sourceVideo.srcObject = this.mediaStream;
        await this.elements.sourceVideo.play();

        console.log('Webcam ready:', this.mediaStream.getVideoTracks()[0].getSettings());
    }

    /**
     * Setup sample video with canvas capture
     * Since captureStream() on cross-origin videos doesn't work,
     * we draw to canvas and capture from there
     */
    async _setupSampleVideo(source) {
        console.log('Setting up sample video...');

        const url = this.sampleVideos[source];

        // Create hidden video element
        const video = document.createElement('video');
        video.crossOrigin = 'anonymous';
        video.src = url;
        video.loop = true;
        video.muted = true;

        await new Promise((resolve, reject) => {
            video.onloadeddata = resolve;
            video.onerror = () => reject(new Error('Failed to load video. Try using Webcam instead.'));
            video.load();
        });

        await video.play();

        // Create canvas to capture video frames
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        // Draw video to canvas continuously
        const drawFrame = () => {
            if (video.paused || video.ended) return;
            ctx.drawImage(video, 0, 0);
            requestAnimationFrame(drawFrame);
        };
        drawFrame();

        // Capture stream from canvas
        this.mediaStream = canvas.captureStream(30);

        // Store video reference for cleanup
        this._sampleVideo = video;
        this._sampleCanvas = canvas;

        // Show in source preview
        this.elements.sourceVideo.srcObject = this.mediaStream;
        await this.elements.sourceVideo.play();

        console.log('Sample video ready:', video.videoWidth, 'x', video.videoHeight);
    }

    /**
     * Create WebRTC loopback connection
     */
    async _createWebRTCLoopback() {
        console.log('Creating WebRTC loopback...');

        // Create peer connections
        this.senderPC = new RTCPeerConnection();
        this.receiverPC = new RTCPeerConnection();

        // Handle ICE candidates
        this.senderPC.onicecandidate = (e) => {
            if (e.candidate) {
                this.receiverPC.addIceCandidate(e.candidate).catch(console.error);
            }
        };

        this.receiverPC.onicecandidate = (e) => {
            if (e.candidate) {
                this.senderPC.addIceCandidate(e.candidate).catch(console.error);
            }
        };

        // Handle incoming track on receiver
        this.receiverPC.ontrack = (e) => {
            console.log('Received track:', e.track.kind);
            if (e.streams && e.streams[0]) {
                this.elements.receivedVideo.srcObject = e.streams[0];
                this.elements.receivedVideo.play().catch(console.error);
            }
        };

        // Connection state logging
        this.senderPC.onconnectionstatechange = () => {
            console.log('Sender connection state:', this.senderPC.connectionState);
        };
        this.receiverPC.onconnectionstatechange = () => {
            console.log('Receiver connection state:', this.receiverPC.connectionState);
        };

        // Add tracks to sender
        const tracks = this.mediaStream.getTracks();
        console.log('Adding tracks:', tracks.length);
        tracks.forEach(track => {
            this.senderPC.addTrack(track, this.mediaStream);
        });

        // Create offer
        const offer = await this.senderPC.createOffer();

        // Modify SDP to prefer selected codec
        const preferredCodec = this.elements.videoCodec.value;
        offer.sdp = this._setPreferredCodec(offer.sdp, preferredCodec);

        await this.senderPC.setLocalDescription(offer);
        await this.receiverPC.setRemoteDescription(offer);

        // Create answer
        const answer = await this.receiverPC.createAnswer();
        await this.receiverPC.setLocalDescription(answer);
        await this.senderPC.setRemoteDescription(answer);

        // Wait for connection
        await this._waitForConnection();

        // Set bitrate
        await this._updateBitrate();

        // Update codec info
        this.elements.codecInfo.textContent = preferredCodec;

        console.log('WebRTC loopback established');
    }

    /**
     * Wait for WebRTC connection to establish
     */
    _waitForConnection() {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Connection timeout')), 10000);

            const checkState = () => {
                if (this.receiverPC.connectionState === 'connected') {
                    clearTimeout(timeout);
                    resolve();
                } else if (this.receiverPC.connectionState === 'failed') {
                    clearTimeout(timeout);
                    reject(new Error('Connection failed'));
                }
            };

            this.receiverPC.onconnectionstatechange = checkState;
            checkState();
        });
    }

    /**
     * Set preferred video codec in SDP
     */
    _setPreferredCodec(sdp, codec) {
        const lines = sdp.split('\r\n');
        let mLineIndex = -1;
        let codecPayload = null;

        // Find video m-line and codec payload
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].startsWith('m=video')) {
                mLineIndex = i;
            }
            if (mLineIndex !== -1 && lines[i].includes('a=rtpmap:')) {
                const match = lines[i].match(/a=rtpmap:(\d+)\s+(\w+)/i);
                if (match && match[2].toUpperCase() === codec.toUpperCase()) {
                    codecPayload = match[1];
                    break;
                }
            }
        }

        if (mLineIndex !== -1 && codecPayload) {
            const mLineParts = lines[mLineIndex].split(' ');
            const payloads = mLineParts.slice(3);
            const newPayloads = [codecPayload, ...payloads.filter(p => p !== codecPayload)];
            lines[mLineIndex] = mLineParts.slice(0, 3).concat(newPayloads).join(' ');
        }

        return lines.join('\r\n');
    }

    /**
     * Update bitrate constraint
     */
    async _updateBitrate() {
        if (!this.senderPC) return;

        const bitrate = parseInt(this.elements.targetBitrate.value) * 1000;

        const senders = this.senderPC.getSenders();
        for (const sender of senders) {
            if (sender.track?.kind === 'video') {
                const params = sender.getParameters();
                if (!params.encodings || params.encodings.length === 0) {
                    params.encodings = [{}];
                }
                params.encodings[0].maxBitrate = bitrate;
                try {
                    await sender.setParameters(params);
                    console.log('Bitrate set to:', bitrate / 1000, 'kbps');
                } catch (e) {
                    console.warn('Failed to set bitrate:', e);
                }
            }
        }
    }

    /**
     * Start WebRTC stats monitoring
     */
    _startStatsMonitoring() {
        this.lastBytesReceived = 0;
        this.lastStatsTime = performance.now();

        this.statsInterval = setInterval(async () => {
            if (!this.receiverPC) return;

            try {
                const stats = await this.receiverPC.getStats();
                let bytesReceived = 0;
                let packetsLost = 0;
                let packetsReceived = 0;

                stats.forEach(report => {
                    if (report.type === 'inbound-rtp' && report.kind === 'video') {
                        bytesReceived = report.bytesReceived || 0;
                        packetsLost = report.packetsLost || 0;
                        packetsReceived = report.packetsReceived || 0;
                    }
                });

                // Calculate actual bitrate
                const now = performance.now();
                const timeDiff = (now - this.lastStatsTime) / 1000;
                const bytesDiff = bytesReceived - this.lastBytesReceived;
                const bitrateKbps = Math.round((bytesDiff * 8) / timeDiff / 1000);

                this.lastBytesReceived = bytesReceived;
                this.lastStatsTime = now;

                if (bitrateKbps > 0) {
                    this.elements.bitrateInfo.textContent = `${bitrateKbps} kbps`;
                }

                // Packet loss
                const totalPackets = packetsReceived + packetsLost;
                const lossPercent = totalPackets > 0 ? ((packetsLost / totalPackets) * 100).toFixed(1) : 0;
                this.elements.packetLoss.textContent = `${lossPercent}%`;

            } catch (e) {
                console.warn('Stats error:', e);
            }
        }, 1000);
    }

    /**
     * Stop everything
     */
    stop() {
        this._stopProcessing();

        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }

        if (this.senderPC) {
            this.senderPC.close();
            this.senderPC = null;
        }
        if (this.receiverPC) {
            this.receiverPC.close();
            this.receiverPC = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Cleanup sample video elements
        if (this._sampleVideo) {
            this._sampleVideo.pause();
            this._sampleVideo.src = '';
            this._sampleVideo = null;
        }
        if (this._sampleCanvas) {
            this._sampleCanvas = null;
        }

        this.elements.sourceVideo.srcObject = null;
        this.elements.receivedVideo.srcObject = null;

        const ctx = this.elements.enhancedCanvas.getContext('2d');
        ctx.clearRect(0, 0, this.elements.enhancedCanvas.width, this.elements.enhancedCanvas.height);

        this.isRunning = false;
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this.elements.originalOverlay.classList.remove('hidden');
        this.elements.enhancedOverlay.classList.remove('hidden');

        this._updateConnectionStatus('disconnected');
        this._resetStats();

        this._showNotification('Stopped', 'info');
    }

    /**
     * Start frame processing
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
     * Process frame
     */
    async _processFrame() {
        if (!this.isRunning) return;

        const startTime = performance.now();

        try {
            const video = this.elements.receivedVideo;

            if (video.readyState >= 2 && video.videoWidth > 0) {
                const inputWidth = video.videoWidth;
                const inputHeight = video.videoHeight;

                let outputWidth = inputWidth;
                let outputHeight = inputHeight;

                const mode = this.elements.enhancementMode.value;
                if (this.elements.enableEnhancement.checked && mode === 'superres') {
                    outputWidth = Math.min(inputWidth * 2, 1920);
                    outputHeight = Math.min(inputHeight * 2, 1080);
                }

                await this.enhancer.processFrame(video, outputWidth, outputHeight);

                this.elements.originalRes.textContent = `${inputWidth}x${inputHeight}`;
                this.elements.enhancedRes.textContent = `${outputWidth}x${outputHeight}`;
                this.elements.inputRes.textContent = `${inputWidth}x${inputHeight}`;
                this.elements.outputRes.textContent = `${outputWidth}x${outputHeight}`;
            }
        } catch (error) {
            console.error('Frame error:', error);
        }

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

        this.animationFrameId = requestAnimationFrame(() => this._processFrame());
    }

    /**
     * Update connection status UI
     */
    _updateConnectionStatus(status) {
        const dot = this.elements.connectionDot;
        const text = this.elements.connectionStatus;

        dot.classList.remove('connected');

        switch (status) {
            case 'connected':
                dot.classList.add('connected');
                text.textContent = 'Connected';
                break;
            case 'connecting':
                text.textContent = 'Connecting...';
                break;
            default:
                text.textContent = 'Disconnected';
                this.elements.codecInfo.textContent = '--';
                this.elements.bitrateInfo.textContent = '--';
                this.elements.packetLoss.textContent = '--';
        }
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
     * Reset stats
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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.app = new WebRTCEnhancementApp();
    window.app.init();
});

window.addEventListener('beforeunload', () => {
    if (window.app) {
        window.app.dispose();
    }
});
