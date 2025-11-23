/**
 * Video Enhancer Module
 * Provides real-time video enhancement using TensorFlow.js and WebGL shaders
 */

class VideoEnhancer {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.glCanvas = null;
        this.gl = null;
        this.programs = {};
        this.textures = {};
        this.framebuffers = {};
        this.initialized = false;
        this.backend = 'webgl';

        // Enhancement settings
        this.settings = {
            mode: 'sharpen',
            intensity: 0.5,
            brightness: 0,
            contrast: 0,
            saturation: 0
        };

        // Super resolution model (will be loaded on demand)
        this.srModel = null;
        this.modelLoading = false;
    }

    /**
     * Initialize the enhancer with a canvas element
     */
    async init(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { willReadFrequently: true });

        // Create WebGL canvas for GPU-accelerated processing
        this.glCanvas = document.createElement('canvas');
        this.gl = this.glCanvas.getContext('webgl2') || this.glCanvas.getContext('webgl');

        if (this.gl) {
            this._initWebGL();
            this.backend = 'webgl';
        } else {
            console.warn('WebGL not available, falling back to Canvas 2D');
            this.backend = 'canvas2d';
        }

        // Initialize TensorFlow.js
        await tf.ready();
        console.log('TensorFlow.js backend:', tf.getBackend());

        this.initialized = true;
        return this.backend;
    }

    /**
     * Initialize WebGL shaders and programs
     */
    _initWebGL() {
        const gl = this.gl;

        // Vertex shader (shared)
        const vertexShaderSource = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            varying vec2 v_texCoord;
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }
        `;

        // Fragment shaders for different enhancement modes
        const shaders = {
            passthrough: `
                precision mediump float;
                varying vec2 v_texCoord;
                uniform sampler2D u_image;
                void main() {
                    gl_FragColor = texture2D(u_image, v_texCoord);
                }
            `,
            sharpen: `
                precision mediump float;
                varying vec2 v_texCoord;
                uniform sampler2D u_image;
                uniform vec2 u_textureSize;
                uniform float u_intensity;

                void main() {
                    vec2 onePixel = vec2(1.0) / u_textureSize;
                    vec4 color = texture2D(u_image, v_texCoord);

                    // Unsharp mask kernel
                    vec4 sum = vec4(0.0);
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2(-1, -1)) * -0.5;
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2( 0, -1)) * -1.0;
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2( 1, -1)) * -0.5;
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2(-1,  0)) * -1.0;
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2( 0,  0)) * 7.0;
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2( 1,  0)) * -1.0;
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2(-1,  1)) * -0.5;
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2( 0,  1)) * -1.0;
                    sum += texture2D(u_image, v_texCoord + onePixel * vec2( 1,  1)) * -0.5;

                    vec4 sharpened = mix(color, sum, u_intensity);
                    gl_FragColor = clamp(sharpened, 0.0, 1.0);
                }
            `,
            denoise: `
                precision mediump float;
                varying vec2 v_texCoord;
                uniform sampler2D u_image;
                uniform vec2 u_textureSize;
                uniform float u_intensity;

                void main() {
                    vec2 onePixel = vec2(1.0) / u_textureSize;

                    // Bilateral filter approximation
                    vec4 sum = vec4(0.0);
                    float totalWeight = 0.0;
                    vec4 centerColor = texture2D(u_image, v_texCoord);

                    for (int x = -2; x <= 2; x++) {
                        for (int y = -2; y <= 2; y++) {
                            vec2 offset = vec2(float(x), float(y)) * onePixel;
                            vec4 sampleColor = texture2D(u_image, v_texCoord + offset);

                            float spatialWeight = exp(-float(x*x + y*y) / 4.0);
                            float colorDiff = length(sampleColor.rgb - centerColor.rgb);
                            float rangeWeight = exp(-colorDiff * colorDiff * 10.0);
                            float weight = spatialWeight * rangeWeight;

                            sum += sampleColor * weight;
                            totalWeight += weight;
                        }
                    }

                    vec4 denoised = sum / totalWeight;
                    gl_FragColor = mix(centerColor, denoised, u_intensity);
                }
            `,
            enhance: `
                precision mediump float;
                varying vec2 v_texCoord;
                uniform sampler2D u_image;
                uniform float u_brightness;
                uniform float u_contrast;
                uniform float u_saturation;

                void main() {
                    vec4 color = texture2D(u_image, v_texCoord);

                    // Brightness
                    color.rgb += u_brightness;

                    // Contrast
                    color.rgb = (color.rgb - 0.5) * (1.0 + u_contrast) + 0.5;

                    // Saturation
                    float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                    color.rgb = mix(vec3(gray), color.rgb, 1.0 + u_saturation);

                    gl_FragColor = clamp(color, 0.0, 1.0);
                }
            `,
            hdr: `
                precision mediump float;
                varying vec2 v_texCoord;
                uniform sampler2D u_image;
                uniform vec2 u_textureSize;
                uniform float u_intensity;

                void main() {
                    vec4 color = texture2D(u_image, v_texCoord);

                    // Local tone mapping (simplified HDR effect)
                    float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));

                    // Compute local average
                    vec2 onePixel = vec2(1.0) / u_textureSize;
                    float localAvg = 0.0;
                    for (int x = -3; x <= 3; x++) {
                        for (int y = -3; y <= 3; y++) {
                            vec4 sample = texture2D(u_image, v_texCoord + vec2(float(x), float(y)) * onePixel * 2.0);
                            localAvg += dot(sample.rgb, vec3(0.299, 0.587, 0.114));
                        }
                    }
                    localAvg /= 49.0;

                    // Adaptive tone mapping
                    float targetLum = 0.5;
                    float adjustment = targetLum / (localAvg + 0.001);
                    adjustment = clamp(adjustment, 0.5, 2.0);

                    vec3 enhanced = color.rgb * mix(1.0, adjustment, u_intensity);

                    // Boost saturation slightly
                    float gray = dot(enhanced, vec3(0.299, 0.587, 0.114));
                    enhanced = mix(vec3(gray), enhanced, 1.0 + u_intensity * 0.3);

                    gl_FragColor = vec4(clamp(enhanced, 0.0, 1.0), color.a);
                }
            `
        };

        // Compile and link programs
        const vertexShader = this._compileShader(gl.VERTEX_SHADER, vertexShaderSource);

        for (const [name, fragSource] of Object.entries(shaders)) {
            const fragmentShader = this._compileShader(gl.FRAGMENT_SHADER, fragSource);
            const program = this._createProgram(vertexShader, fragmentShader);
            this.programs[name] = {
                program,
                locations: this._getLocations(program)
            };
        }

        // Set up geometry
        this._setupGeometry();
    }

    _compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    _createProgram(vertexShader, fragmentShader) {
        const gl = this.gl;
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(program));
            return null;
        }
        return program;
    }

    _getLocations(program) {
        const gl = this.gl;
        return {
            position: gl.getAttribLocation(program, 'a_position'),
            texCoord: gl.getAttribLocation(program, 'a_texCoord'),
            image: gl.getUniformLocation(program, 'u_image'),
            textureSize: gl.getUniformLocation(program, 'u_textureSize'),
            intensity: gl.getUniformLocation(program, 'u_intensity'),
            brightness: gl.getUniformLocation(program, 'u_brightness'),
            contrast: gl.getUniformLocation(program, 'u_contrast'),
            saturation: gl.getUniformLocation(program, 'u_saturation')
        };
    }

    _setupGeometry() {
        const gl = this.gl;

        // Position buffer
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 1, -1, -1, 1,
            -1, 1, 1, -1, 1, 1
        ]), gl.STATIC_DRAW);
        this.positionBuffer = positionBuffer;

        // Texture coordinate buffer
        const texCoordBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            0, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 0
        ]), gl.STATIC_DRAW);
        this.texCoordBuffer = texCoordBuffer;

        // Create texture
        this.texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    }

    /**
     * Update enhancement settings
     */
    updateSettings(settings) {
        this.settings = { ...this.settings, ...settings };
    }

    /**
     * Process a video frame
     */
    async processFrame(source, outputWidth, outputHeight) {
        if (!this.initialized) {
            throw new Error('Enhancer not initialized');
        }

        const sourceWidth = source.videoWidth || source.width;
        const sourceHeight = source.videoHeight || source.height;

        // Handle super resolution mode
        if (this.settings.mode === 'superres') {
            return this._processSuperResolution(source, sourceWidth, sourceHeight, outputWidth, outputHeight);
        }

        // Use WebGL for other modes
        if (this.gl) {
            return this._processWebGL(source, sourceWidth, sourceHeight, outputWidth, outputHeight);
        }

        // Fallback to Canvas 2D
        return this._processCanvas2D(source, sourceWidth, sourceHeight, outputWidth, outputHeight);
    }

    /**
     * Process frame using WebGL
     */
    _processWebGL(source, sourceWidth, sourceHeight, outputWidth, outputHeight) {
        const gl = this.gl;

        // Resize canvas if needed
        if (this.glCanvas.width !== outputWidth || this.glCanvas.height !== outputHeight) {
            this.glCanvas.width = outputWidth;
            this.glCanvas.height = outputHeight;
        }

        // Upload texture
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

        // Select program based on mode
        let programName = this.settings.mode;
        if (!this.programs[programName]) {
            programName = 'passthrough';
        }

        const { program, locations } = this.programs[programName];
        gl.useProgram(program);

        // Set viewport
        gl.viewport(0, 0, outputWidth, outputHeight);

        // Set attributes
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.enableVertexAttribArray(locations.position);
        gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
        gl.enableVertexAttribArray(locations.texCoord);
        gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

        // Set uniforms
        gl.uniform1i(locations.image, 0);

        if (locations.textureSize) {
            gl.uniform2f(locations.textureSize, sourceWidth, sourceHeight);
        }
        if (locations.intensity !== null) {
            gl.uniform1f(locations.intensity, this.settings.intensity);
        }
        if (locations.brightness !== null) {
            gl.uniform1f(locations.brightness, this.settings.brightness);
        }
        if (locations.contrast !== null) {
            gl.uniform1f(locations.contrast, this.settings.contrast);
        }
        if (locations.saturation !== null) {
            gl.uniform1f(locations.saturation, this.settings.saturation);
        }

        // Draw
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Copy to output canvas
        this.canvas.width = outputWidth;
        this.canvas.height = outputHeight;
        this.ctx.drawImage(this.glCanvas, 0, 0);

        // Apply additional color adjustments if in sharpen/denoise/hdr mode
        if (['sharpen', 'denoise', 'hdr'].includes(this.settings.mode)) {
            this._applyColorAdjustments(outputWidth, outputHeight);
        }

        return this.canvas;
    }

    /**
     * Apply brightness/contrast/saturation adjustments
     */
    _applyColorAdjustments(width, height) {
        if (this.settings.brightness === 0 &&
            this.settings.contrast === 0 &&
            this.settings.saturation === 0) {
            return;
        }

        const imageData = this.ctx.getImageData(0, 0, width, height);
        const data = imageData.data;

        const brightness = this.settings.brightness;
        const contrast = this.settings.contrast + 1;
        const saturation = this.settings.saturation + 1;

        for (let i = 0; i < data.length; i += 4) {
            let r = data[i];
            let g = data[i + 1];
            let b = data[i + 2];

            // Brightness
            r += brightness * 255;
            g += brightness * 255;
            b += brightness * 255;

            // Contrast
            r = ((r / 255 - 0.5) * contrast + 0.5) * 255;
            g = ((g / 255 - 0.5) * contrast + 0.5) * 255;
            b = ((b / 255 - 0.5) * contrast + 0.5) * 255;

            // Saturation
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            r = gray + saturation * (r - gray);
            g = gray + saturation * (g - gray);
            b = gray + saturation * (b - gray);

            // Clamp
            data[i] = Math.max(0, Math.min(255, r));
            data[i + 1] = Math.max(0, Math.min(255, g));
            data[i + 2] = Math.max(0, Math.min(255, b));
        }

        this.ctx.putImageData(imageData, 0, 0);
    }

    /**
     * Process frame using Canvas 2D (fallback)
     */
    _processCanvas2D(source, sourceWidth, sourceHeight, outputWidth, outputHeight) {
        this.canvas.width = outputWidth;
        this.canvas.height = outputHeight;

        // Draw source
        this.ctx.drawImage(source, 0, 0, outputWidth, outputHeight);

        // Apply simple sharpening using convolution
        if (this.settings.mode === 'sharpen' && this.settings.intensity > 0) {
            const imageData = this.ctx.getImageData(0, 0, outputWidth, outputHeight);
            const sharpened = this._sharpenKernel(imageData, this.settings.intensity);
            this.ctx.putImageData(sharpened, 0, 0);
        }

        // Apply color adjustments
        this._applyColorAdjustments(outputWidth, outputHeight);

        return this.canvas;
    }

    /**
     * Simple sharpening kernel for Canvas 2D fallback
     */
    _sharpenKernel(imageData, intensity) {
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        const output = new ImageData(width, height);
        const outData = output.data;

        // Sharpening kernel
        const kernel = [
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
        ];

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                for (let c = 0; c < 3; c++) {
                    let sum = 0;
                    for (let ky = -1; ky <= 1; ky++) {
                        for (let kx = -1; kx <= 1; kx++) {
                            const idx = ((y + ky) * width + (x + kx)) * 4 + c;
                            sum += data[idx] * kernel[(ky + 1) * 3 + (kx + 1)];
                        }
                    }
                    const idx = (y * width + x) * 4 + c;
                    const original = data[idx];
                    outData[idx] = Math.max(0, Math.min(255,
                        original + (sum - original) * intensity));
                }
                outData[(y * width + x) * 4 + 3] = data[(y * width + x) * 4 + 3];
            }
        }

        return output;
    }

    /**
     * Super Resolution using TensorFlow.js
     * Uses a simple bicubic + sharpening approach for real-time performance
     */
    async _processSuperResolution(source, sourceWidth, sourceHeight, outputWidth, outputHeight) {
        // For real-time performance, we use enhanced bicubic interpolation
        // True ML super resolution models are too slow for real-time video

        const scale = 2;
        const targetWidth = Math.min(sourceWidth * scale, outputWidth);
        const targetHeight = Math.min(sourceHeight * scale, outputHeight);

        this.canvas.width = outputWidth;
        this.canvas.height = outputHeight;

        // Draw upscaled
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
        this.ctx.drawImage(source, 0, 0, targetWidth, targetHeight);

        // Apply sharpening to enhance details
        if (this.settings.intensity > 0) {
            const imageData = this.ctx.getImageData(0, 0, targetWidth, targetHeight);

            // Use TensorFlow.js for faster processing
            const tensor = tf.browser.fromPixels(imageData);

            // Apply Laplacian sharpening
            const sharpened = tf.tidy(() => {
                const floatTensor = tensor.toFloat().div(255);

                // Laplacian kernel for edge enhancement
                const kernel = tf.tensor4d([
                    [[[-1]], [[-1]], [[-1]]],
                    [[[-1]], [[9]], [[-1]]],
                    [[[-1]], [[-1]], [[-1]]]
                ], [3, 3, 1, 1]);

                // Process each channel
                const channels = tf.split(floatTensor, 3, 2);
                const enhanced = channels.map(ch => {
                    const expanded = ch.expandDims(0);
                    const conv = tf.conv2d(expanded, kernel, 1, 'same');
                    return conv.squeeze([0]);
                });

                const merged = tf.stack(enhanced, 2).squeeze();
                const intensity = this.settings.intensity;
                const blended = floatTensor.mul(1 - intensity).add(merged.mul(intensity));

                return blended.clipByValue(0, 1).mul(255).toInt();
            });

            const resultData = await tf.browser.toPixels(sharpened);
            const outputImageData = new ImageData(
                new Uint8ClampedArray(resultData),
                targetWidth,
                targetHeight
            );
            this.ctx.putImageData(outputImageData, 0, 0);

            sharpened.dispose();
            tensor.dispose();
        }

        // Apply color adjustments
        this._applyColorAdjustments(targetWidth, targetHeight);

        return this.canvas;
    }

    /**
     * Get current backend info
     */
    getBackendInfo() {
        return {
            enhancer: this.backend,
            tensorflow: tf.getBackend()
        };
    }

    /**
     * Dispose resources
     */
    dispose() {
        if (this.gl) {
            // Clean up WebGL resources
            for (const { program } of Object.values(this.programs)) {
                this.gl.deleteProgram(program);
            }
            if (this.texture) {
                this.gl.deleteTexture(this.texture);
            }
        }

        if (this.srModel) {
            this.srModel.dispose();
        }

        this.initialized = false;
    }
}

// Export for use in other modules
window.VideoEnhancer = VideoEnhancer;
