/**
 * WebRTC Compression Loopback Module
 * Creates a local WebRTC loopback to actually encode/decode video
 * This allows comparing original vs compressed video side by side
 */

class CompressionLoopback {
    constructor() {
        this.senderPC = null;
        this.receiverPC = null;
        this.localStream = null;
        this.compressedStream = null;

        // Configuration
        this.config = {
            codec: 'VP8',  // VP8, VP9, H264
            bitrate: 1000, // kbps
            framerate: 30,
            width: 1280,
            height: 720
        };

        // Stats tracking
        this.stats = {
            bytesSent: 0,
            bytesReceived: 0,
            packetsLost: 0,
            jitter: 0,
            roundTripTime: 0,
            actualBitrate: 0,
            frameWidth: 0,
            frameHeight: 0,
            framesPerSecond: 0,
            framesDropped: 0,
            codecName: ''
        };

        this.statsInterval = null;
        this.onStats = null;
        this.onCompressedStream = null;
    }

    /**
     * Get supported codecs
     */
    static getSupportedCodecs() {
        const codecs = [
            { id: 'VP8', name: 'VP8', mimeType: 'video/VP8' },
            { id: 'VP9', name: 'VP9', mimeType: 'video/VP9' },
            { id: 'H264', name: 'H.264', mimeType: 'video/H264' }
        ];

        // Check for AV1 support
        if (RTCRtpSender.getCapabilities) {
            const capabilities = RTCRtpSender.getCapabilities('video');
            if (capabilities && capabilities.codecs.some(c => c.mimeType.includes('AV1'))) {
                codecs.push({ id: 'AV1', name: 'AV1', mimeType: 'video/AV1' });
            }
        }

        return codecs;
    }

    /**
     * Update compression configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };

        // Apply new bitrate if connection exists
        if (this.senderPC) {
            this._applyBitrateConstraints();
        }
    }

    /**
     * Start the compression loopback with a media stream
     */
    async start(sourceStream) {
        await this.stop();

        this.localStream = sourceStream;

        // Create peer connections
        const pcConfig = {
            iceServers: [],  // Local loopback doesn't need ICE servers
            encodedInsertableStreams: false
        };

        this.senderPC = new RTCPeerConnection(pcConfig);
        this.receiverPC = new RTCPeerConnection(pcConfig);

        // Exchange ICE candidates
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

        // Handle incoming stream on receiver
        this.receiverPC.ontrack = (e) => {
            console.log('Received compressed stream');
            this.compressedStream = e.streams[0];
            if (this.onCompressedStream) {
                this.onCompressedStream(this.compressedStream);
            }
        };

        // Add tracks to sender with encoding parameters
        const videoTrack = sourceStream.getVideoTracks()[0];
        if (videoTrack) {
            const sender = this.senderPC.addTrack(videoTrack, sourceStream);
            await this._configureEncodingParameters(sender);
        }

        // Create and exchange offer/answer
        const offer = await this.senderPC.createOffer();

        // Modify SDP for codec preference
        const modifiedOffer = this._setPreferredCodec(offer.sdp, this.config.codec);
        await this.senderPC.setLocalDescription({ type: 'offer', sdp: modifiedOffer });

        await this.receiverPC.setRemoteDescription({ type: 'offer', sdp: modifiedOffer });

        const answer = await this.receiverPC.createAnswer();
        await this.receiverPC.setLocalDescription(answer);
        await this.senderPC.setRemoteDescription(answer);

        // Start stats collection
        this._startStatsCollection();

        console.log(`Compression loopback started: ${this.config.codec} @ ${this.config.bitrate}kbps`);

        return this.compressedStream;
    }

    /**
     * Configure encoding parameters on the sender
     */
    async _configureEncodingParameters(sender) {
        const params = sender.getParameters();

        if (!params.encodings || params.encodings.length === 0) {
            params.encodings = [{}];
        }

        params.encodings[0].maxBitrate = this.config.bitrate * 1000; // Convert to bps
        params.encodings[0].maxFramerate = this.config.framerate;

        // Scale down if requested
        if (this.config.scaleResolutionDownBy) {
            params.encodings[0].scaleResolutionDownBy = this.config.scaleResolutionDownBy;
        }

        try {
            await sender.setParameters(params);
            console.log('Encoding parameters set:', params.encodings[0]);
        } catch (e) {
            console.warn('Could not set encoding parameters:', e);
        }
    }

    /**
     * Apply bitrate constraints to existing sender
     */
    async _applyBitrateConstraints() {
        const senders = this.senderPC.getSenders();
        const videoSender = senders.find(s => s.track && s.track.kind === 'video');

        if (videoSender) {
            await this._configureEncodingParameters(videoSender);
        }
    }

    /**
     * Modify SDP to prefer a specific codec
     */
    _setPreferredCodec(sdp, codecName) {
        const lines = sdp.split('\r\n');
        const mLineIndex = lines.findIndex(line => line.startsWith('m=video'));

        if (mLineIndex === -1) return sdp;

        // Find payload type for the preferred codec
        let preferredPayloadType = null;
        const codecPattern = new RegExp(`a=rtpmap:(\\d+)\\s+${codecName}/`, 'i');

        for (const line of lines) {
            const match = line.match(codecPattern);
            if (match) {
                preferredPayloadType = match[1];
                break;
            }
        }

        if (!preferredPayloadType) {
            console.warn(`Codec ${codecName} not found in SDP`);
            return sdp;
        }

        // Reorder m-line to prefer this codec
        const mLine = lines[mLineIndex];
        const parts = mLine.split(' ');
        const payloadTypes = parts.slice(3);

        // Move preferred codec to front
        const newPayloadTypes = [preferredPayloadType, ...payloadTypes.filter(pt => pt !== preferredPayloadType)];
        parts.splice(3, payloadTypes.length, ...newPayloadTypes);
        lines[mLineIndex] = parts.join(' ');

        return lines.join('\r\n');
    }

    /**
     * Start collecting WebRTC stats
     */
    _startStatsCollection() {
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
        }

        let lastBytesSent = 0;
        let lastTimestamp = Date.now();

        this.statsInterval = setInterval(async () => {
            if (!this.senderPC || !this.receiverPC) return;

            try {
                // Get sender stats
                const senderStats = await this.senderPC.getStats();
                const receiverStats = await this.receiverPC.getStats();

                senderStats.forEach(report => {
                    if (report.type === 'outbound-rtp' && report.kind === 'video') {
                        const now = Date.now();
                        const elapsed = (now - lastTimestamp) / 1000;

                        if (elapsed > 0 && report.bytesSent > lastBytesSent) {
                            const bitrate = ((report.bytesSent - lastBytesSent) * 8) / elapsed / 1000;
                            this.stats.actualBitrate = Math.round(bitrate);
                        }

                        lastBytesSent = report.bytesSent;
                        lastTimestamp = now;

                        this.stats.bytesSent = report.bytesSent;
                        this.stats.framesPerSecond = report.framesPerSecond || 0;
                        this.stats.frameWidth = report.frameWidth || 0;
                        this.stats.frameHeight = report.frameHeight || 0;
                    }

                    if (report.type === 'codec' && report.mimeType && report.mimeType.includes('video')) {
                        this.stats.codecName = report.mimeType.split('/')[1];
                    }

                    if (report.type === 'candidate-pair' && report.state === 'succeeded') {
                        this.stats.roundTripTime = report.currentRoundTripTime ? Math.round(report.currentRoundTripTime * 1000) : 0;
                    }
                });

                receiverStats.forEach(report => {
                    if (report.type === 'inbound-rtp' && report.kind === 'video') {
                        this.stats.bytesReceived = report.bytesReceived;
                        this.stats.packetsLost = report.packetsLost || 0;
                        this.stats.jitter = report.jitter ? Math.round(report.jitter * 1000) : 0;
                        this.stats.framesDropped = report.framesDropped || 0;
                    }
                });

                if (this.onStats) {
                    this.onStats(this.stats);
                }
            } catch (e) {
                console.warn('Stats collection error:', e);
            }
        }, 1000);
    }

    /**
     * Get the compressed stream
     */
    getCompressedStream() {
        return this.compressedStream;
    }

    /**
     * Get current stats
     */
    getStats() {
        return { ...this.stats };
    }

    /**
     * Stop the loopback
     */
    async stop() {
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

        this.compressedStream = null;

        // Reset stats
        this.stats = {
            bytesSent: 0,
            bytesReceived: 0,
            packetsLost: 0,
            jitter: 0,
            roundTripTime: 0,
            actualBitrate: 0,
            frameWidth: 0,
            frameHeight: 0,
            framesPerSecond: 0,
            framesDropped: 0,
            codecName: ''
        };
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.stop();
        this.localStream = null;
        this.onStats = null;
        this.onCompressedStream = null;
    }
}

// Export for use in other modules
window.CompressionLoopback = CompressionLoopback;
