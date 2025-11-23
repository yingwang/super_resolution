/**
 * Background Segmentation Module
 * Uses TensorFlow.js BodyPix for person segmentation
 */

class BackgroundSegmentation {
    constructor() {
        this.model = null;
        this.isLoading = false;
        this.isEnabled = false;

        // Segmentation settings
        this.settings = {
            backgroundType: 'blur', // 'blur', 'color', 'image', 'transparent'
            blurAmount: 10,
            backgroundColor: '#00ff00',
            backgroundImage: null,
            edgeBlur: 3,
            flipHorizontal: false
        };

        // Offscreen canvases for processing
        this.segCanvas = null;
        this.segCtx = null;
        this.outputCanvas = null;
        this.outputCtx = null;
    }

    /**
     * Initialize and load the segmentation model
     */
    async init() {
        if (this.model || this.isLoading) return;

        this.isLoading = true;
        console.log('Loading BodyPix model...');

        try {
            // Load BodyPix model
            this.model = await bodyPix.load({
                architecture: 'MobileNetV1',
                outputStride: 16,
                multiplier: 0.75,
                quantBytes: 2
            });

            console.log('BodyPix model loaded');
            this.isLoading = false;
            return true;
        } catch (error) {
            console.error('Failed to load BodyPix:', error);
            this.isLoading = false;
            throw error;
        }
    }

    /**
     * Update segmentation settings
     */
    updateSettings(settings) {
        this.settings = { ...this.settings, ...settings };
    }

    /**
     * Enable/disable segmentation
     */
    setEnabled(enabled) {
        this.isEnabled = enabled;
    }

    /**
     * Process a video frame with background segmentation
     */
    async processFrame(sourceVideo, outputCanvas) {
        if (!this.model || !this.isEnabled) {
            return false;
        }

        const width = sourceVideo.videoWidth;
        const height = sourceVideo.videoHeight;

        if (width === 0 || height === 0) return false;

        // Ensure output canvas is properly sized
        if (outputCanvas.width !== width || outputCanvas.height !== height) {
            outputCanvas.width = width;
            outputCanvas.height = height;
        }

        // Create offscreen canvas if needed
        if (!this.segCanvas || this.segCanvas.width !== width) {
            this.segCanvas = document.createElement('canvas');
            this.segCanvas.width = width;
            this.segCanvas.height = height;
            this.segCtx = this.segCanvas.getContext('2d');
        }

        const ctx = outputCanvas.getContext('2d');

        try {
            // Perform segmentation
            const segmentation = await this.model.segmentPerson(sourceVideo, {
                flipHorizontal: this.settings.flipHorizontal,
                internalResolution: 'medium',
                segmentationThreshold: 0.7
            });

            // Apply background effect based on type
            switch (this.settings.backgroundType) {
                case 'blur':
                    await this._applyBlurBackground(sourceVideo, segmentation, ctx, width, height);
                    break;
                case 'color':
                    this._applyColorBackground(sourceVideo, segmentation, ctx, width, height);
                    break;
                case 'transparent':
                    this._applyTransparentBackground(sourceVideo, segmentation, ctx, width, height);
                    break;
                case 'image':
                    this._applyImageBackground(sourceVideo, segmentation, ctx, width, height);
                    break;
                default:
                    ctx.drawImage(sourceVideo, 0, 0, width, height);
            }

            return true;
        } catch (error) {
            console.error('Segmentation error:', error);
            // Fallback: just draw the original video
            ctx.drawImage(sourceVideo, 0, 0, width, height);
            return false;
        }
    }

    /**
     * Apply blur to background
     */
    async _applyBlurBackground(source, segmentation, ctx, width, height) {
        // Draw blurred background
        this.segCtx.filter = `blur(${this.settings.blurAmount}px)`;
        this.segCtx.drawImage(source, 0, 0, width, height);
        this.segCtx.filter = 'none';

        // Draw blurred image to output
        ctx.drawImage(this.segCanvas, 0, 0);

        // Create person mask
        const mask = this._createMask(segmentation, width, height);

        // Draw original video only where person is
        ctx.save();
        ctx.globalCompositeOperation = 'destination-out';
        ctx.putImageData(mask, 0, 0);
        ctx.globalCompositeOperation = 'destination-over';
        ctx.drawImage(source, 0, 0, width, height);
        ctx.restore();
    }

    /**
     * Apply solid color background
     */
    _applyColorBackground(source, segmentation, ctx, width, height) {
        // Fill with background color
        ctx.fillStyle = this.settings.backgroundColor;
        ctx.fillRect(0, 0, width, height);

        // Draw person on top using mask
        this._drawPersonWithMask(source, segmentation, ctx, width, height);
    }

    /**
     * Apply transparent background
     */
    _applyTransparentBackground(source, segmentation, ctx, width, height) {
        // Clear canvas (transparent)
        ctx.clearRect(0, 0, width, height);

        // Draw person only
        this._drawPersonWithMask(source, segmentation, ctx, width, height);
    }

    /**
     * Apply image background
     */
    _applyImageBackground(source, segmentation, ctx, width, height) {
        // Draw background image or color fallback
        if (this.settings.backgroundImage) {
            ctx.drawImage(this.settings.backgroundImage, 0, 0, width, height);
        } else {
            ctx.fillStyle = this.settings.backgroundColor;
            ctx.fillRect(0, 0, width, height);
        }

        // Draw person on top
        this._drawPersonWithMask(source, segmentation, ctx, width, height);
    }

    /**
     * Draw person with segmentation mask
     */
    _drawPersonWithMask(source, segmentation, ctx, width, height) {
        // Draw source to temp canvas
        this.segCtx.drawImage(source, 0, 0, width, height);

        // Get image data
        const imageData = this.segCtx.getImageData(0, 0, width, height);
        const pixels = imageData.data;

        // Apply mask - make non-person pixels transparent
        for (let i = 0; i < segmentation.data.length; i++) {
            if (segmentation.data[i] === 0) {
                // Not a person - make transparent
                pixels[i * 4 + 3] = 0;
            }
        }

        // Apply edge blur for smoother edges
        if (this.settings.edgeBlur > 0) {
            this._applyEdgeSmoothing(pixels, segmentation.data, width, height);
        }

        ctx.putImageData(imageData, 0, 0);
    }

    /**
     * Create mask ImageData from segmentation
     */
    _createMask(segmentation, width, height) {
        const mask = new ImageData(width, height);
        const pixels = mask.data;

        for (let i = 0; i < segmentation.data.length; i++) {
            const isBackground = segmentation.data[i] === 0;
            pixels[i * 4] = isBackground ? 0 : 255;
            pixels[i * 4 + 1] = isBackground ? 0 : 255;
            pixels[i * 4 + 2] = isBackground ? 0 : 255;
            pixels[i * 4 + 3] = isBackground ? 255 : 0;
        }

        return mask;
    }

    /**
     * Apply edge smoothing to reduce jagged edges
     */
    _applyEdgeSmoothing(pixels, maskData, width, height) {
        const blur = this.settings.edgeBlur;

        for (let y = blur; y < height - blur; y++) {
            for (let x = blur; x < width - blur; x++) {
                const i = y * width + x;

                // Check if this is an edge pixel
                let isEdge = false;
                for (let dy = -1; dy <= 1 && !isEdge; dy++) {
                    for (let dx = -1; dx <= 1 && !isEdge; dx++) {
                        const ni = (y + dy) * width + (x + dx);
                        if (maskData[i] !== maskData[ni]) {
                            isEdge = true;
                        }
                    }
                }

                if (isEdge) {
                    // Average alpha in neighborhood for smoother edge
                    let alphaSum = 0;
                    let count = 0;
                    for (let dy = -blur; dy <= blur; dy++) {
                        for (let dx = -blur; dx <= blur; dx++) {
                            const ni = (y + dy) * width + (x + dx);
                            alphaSum += maskData[ni] === 1 ? 255 : 0;
                            count++;
                        }
                    }
                    pixels[i * 4 + 3] = Math.round(alphaSum / count);
                }
            }
        }
    }

    /**
     * Set background image from URL
     */
    async setBackgroundImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                this.settings.backgroundImage = img;
                resolve(img);
            };
            img.onerror = reject;
            img.src = url;
        });
    }

    /**
     * Check if model is loaded
     */
    isReady() {
        return this.model !== null && !this.isLoading;
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.model = null;
        this.segCanvas = null;
        this.segCtx = null;
        this.isEnabled = false;
    }
}

// Export for use in other modules
window.BackgroundSegmentation = BackgroundSegmentation;
