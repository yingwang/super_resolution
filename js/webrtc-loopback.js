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

        // Sample video URLs
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
            this._showNotification('Ready! Click Start WebRTC to begin.', 'success');
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

            // Get source video stream
            await this._setupSourceStream();

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
            this._showNotification('WebRTC connected! Video is being encoded and decoded.', 'success');

        } catch (error) {
            console.error('Start error:', error);
            this._updateConnectionStatus('disconnected');
            this._showNotification(`Failed: ${error.message}`, 'error');
        }
    }

    /**
     * Setup source video stream
     */
    async _setupSourceStream() {
        const source = this.elements.videoSource.value;

        if (source === 'webcam') {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false
            });
            this.elements.sourceVideo.srcObject = this.mediaStream;
        } else {
            // For sample videos, we need to capture from video element
            const url = this.sampleVideos[source];
            this.elements.sourceVideo.src = url;
            this.elements.sourceVideo.crossOrigin = 'anonymous';
            this.elements.sourceVideo.loop = true;

            await new Promise((resolve, reject) => {
                this.elements.sourceVideo.onloadeddata = resolve;
                this.elements.sourceVideo.onerror = () => reject(new Error('Failed to load video'));
                this.elements.sourceVideo.load();
            });

            await this.elements.sourceVideo.play();

            // Capture stream from video element
            this.mediaStream = this.elements.sourceVideo.captureStream();
        }

        // Show source preview
        if (!this.elements.sourceVideo.srcObject) {
            this.elements.sourceVideo.srcObject = this.mediaStream;
        }
        await this.elements.sourceVideo.play().catch(() => {});
    }

    /**
     * Create WebRTC loopback connection
     */
    async _createWebRTCLoopback() {
        // Create peer connections
        this.senderPC = new RTCPeerConnection({
            iceServers: [] // No ICE servers needed for loopback
        });

        this.receiverPC = new RTCPeerConnection({
            iceServers: []
        });

        // Handle ICE candidates
        this.senderPC.onicecandidate = (e) => {
            if (e.candidate) {
                this.receiverPC.addIceCandidate(e.candidate);
            }
        };

        this.receiverPC.onicecandidate = (e) => {
            if (e.candidate) {
                this.senderPC.addIceCandidate(e.candidate);
            }
        };

        // Handle incoming track on receiver
        this.receiverPC.ontrack = (e) => {
            console.log('Received track:', e.track.kind);
            this.elements.receivedVideo.srcObject = e.streams[0];
        };

        // Add tracks to sender
        this.mediaStream.getTracks().forEach(track => {
            this.senderPC.addTrack(track, this.mediaStream);
        });

        // Create offer with codec preferences
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

        // Set initial bitrate
        await this._updateBitrate();

        // Update codec info
        this.elements.codecInfo.textContent = preferredCodec;
    }

    /**
     * Set preferred video codec in SDP
     */
    _setPreferredCodec(sdp, codec) {
        const codecMap = {
            'VP8': 'VP8',
            'VP9': 'VP9',
            'H264': 'H264'
        };

        const lines = sdp.split('\r\n');
        let mLineIndex = -1;
        let codecPayload = null;

        // Find video m-line and codec payload
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].startsWith('m=video')) {
                mLineIndex = i;
            }
            if (mLineIndex !== -1 && lines[i].includes(`a=rtpmap:`) && lines[i].toLowerCase().includes(codecMap[codec].toLowerCase())) {
                codecPayload = lines[i].split(':')[1].split(' ')[0];
                break;
            }
        }

        if (mLineIndex !== -1 && codecPayload) {
            // Reorder payload types in m-line to prefer our codec
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

        const bitrate = parseInt(this.elements.targetBitrate.value) * 1000; // Convert to bps

        const senders = this.senderPC.getSenders();
        for (const sender of senders) {
            if (sender.track?.kind === 'video') {
                const params = sender.getParameters();
                if (!params.encodings || params.encodings.length === 0) {
                    params.encodings = [{}];
                }
                params.encodings[0].maxBitrate = bitrate;
                await sender.setParameters(params);
            }
        }

        console.log('Bitrate set to:', bitrate / 1000, 'kbps');
    }

    /**
     * Start WebRTC stats monitoring
     */
    _startStatsMonitoring() {
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

                // Calculate bitrate (approximate)
                const bitrateKbps = Math.round((bytesReceived * 8) / 1000 / (performance.now() / 1000));
                this.elements.bitrateInfo.textContent = `~${bitrateKbps} kbps`;

                // Packet loss percentage
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

        // Stop stats monitoring
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }

        // Close peer connections
        if (this.senderPC) {
            this.senderPC.close();
            this.senderPC = null;
        }
        if (this.receiverPC) {
            this.receiverPC.close();
            this.receiverPC = null;
        }

        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Reset videos
        this.elements.sourceVideo.srcObject = null;
        this.elements.sourceVideo.src = '';
        this.elements.receivedVideo.srcObject = null;

        // Clear canvas
        const ctx = this.elements.enhancedCanvas.getContext('2d');
        ctx.clearRect(0, 0, this.elements.enhancedCanvas.width, this.elements.enhancedCanvas.height);

        // Update UI
        this.isRunning = false;
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this.elements.originalOverlay.classList.remove('hidden');
        this.elements.enhancedOverlay.classList.remove('hidden');

        this._updateConnectionStatus('disconnected');
        this._resetStats();

        this._showNotification('WebRTC stopped', 'info');
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

                // Update resolution displays
                this.elements.originalRes.textContent = `${inputWidth}x${inputHeight}`;
                this.elements.enhancedRes.textContent = `${outputWidth}x${outputHeight}`;
                this.elements.inputRes.textContent = `${inputWidth}x${inputHeight}`;
                this.elements.outputRes.textContent = `${outputWidth}x${outputHeight}`;
            }
        } catch (error) {
            console.error('Frame error:', error);
        }

        // Stats
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
