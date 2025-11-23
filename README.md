# WebRTC Video Enhancement POC

A proof-of-concept web application that demonstrates real-time video streaming with local super-resolution and video enhancement using WebRTC, TensorFlow.js, and WebGL shaders.

## Features

- **Multiple Video Sources**
  - Sample videos from Google's public video bucket
  - Custom URL input (MP4, WebM)
  - Webcam capture via WebRTC getUserMedia
  - Screen capture via getDisplayMedia

- **Real-time Video Enhancement**
  - **Sharpen** - Unsharp mask filtering for edge enhancement
  - **Super Resolution (2x)** - Bicubic upscaling with detail enhancement using TensorFlow.js
  - **Denoise** - Bilateral filter approximation for noise reduction
  - **Auto Enhance** - Automatic brightness/contrast/saturation adjustment
  - **HDR Effect** - Local tone mapping for improved dynamic range

- **Adjustable Parameters**
  - Enhancement intensity (0-100%)
  - Brightness (-100 to +100)
  - Contrast (-100 to +100)
  - Saturation (-100 to +100)

- **Performance Optimized**
  - WebGL shaders for GPU-accelerated processing
  - TensorFlow.js with WebGL backend
  - Real-time FPS and frame time monitoring
  - Fallback to Canvas 2D if WebGL unavailable

## Quick Start

### Option 1: Using a static file server

```bash
# Install dependencies
npm install

# Start the development server
npm start
```

Then open http://localhost:3000 in Chrome.

### Option 2: Using Python's built-in server

```bash
# Python 3
python -m http.server 3000

# Python 2
python -m SimpleHTTPServer 3000
```

### Option 3: Open directly in browser

Simply open `index.html` in Chrome (some features like webcam access require HTTPS or localhost).

## Usage

1. **Select Video Source**
   - Choose from sample videos, enter a custom URL, or use your webcam

2. **Configure Enhancement**
   - Enable/disable enhancement
   - Select enhancement mode (Sharpen, Super Resolution, Denoise, etc.)
   - Adjust intensity and color parameters

3. **Playback Controls**
   - Play/Pause, Stop, Fullscreen
   - Seek bar for video files
   - Split view for side-by-side comparison

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space / K | Play/Pause |
| F | Toggle Fullscreen |
| ← / → | Seek -5s / +5s |
| ↑ / ↓ | Increase/Decrease intensity |

## Architecture

```
super_resolution/
├── index.html           # Main HTML page
├── styles.css           # Application styles
├── js/
│   ├── app.js           # Main application logic
│   ├── enhancer.js      # Video enhancement (WebGL shaders + TensorFlow.js)
│   └── video-processor.js # Video streaming (WebRTC, MediaStream)
├── package.json
└── README.md
```

### Key Components

- **VideoEnhancer** (`js/enhancer.js`)
  - WebGL shader-based image processing
  - TensorFlow.js integration for ML operations
  - Multiple enhancement modes with configurable parameters

- **VideoProcessor** (`js/video-processor.js`)
  - WebRTC MediaStream handling
  - Video source management (URL, webcam, screen)
  - Frame processing loop with performance monitoring

- **App** (`js/app.js`)
  - UI event handling
  - Component orchestration
  - Playback controls

## Technical Details

### WebRTC Integration

The app uses the following WebRTC/Media APIs:
- `getUserMedia()` - Webcam access
- `getDisplayMedia()` - Screen capture
- `RTCPeerConnection` - Peer-to-peer streaming capability (ready for extension)

### Enhancement Pipeline

1. Video frame captured from source
2. Frame uploaded to WebGL texture
3. Enhancement shader applied (GPU accelerated)
4. Result rendered to output canvas
5. Performance stats updated

### Super Resolution Approach

For real-time performance, the super resolution mode uses:
1. High-quality bicubic interpolation (2x upscale)
2. Laplacian sharpening via TensorFlow.js convolution
3. Adaptive intensity blending

Note: True deep learning super resolution models (ESRGAN, etc.) are too computationally expensive for real-time video. This implementation prioritizes smooth playback.

## Browser Compatibility

- **Chrome 90+** (recommended) - Full WebGL2 and WebRTC support
- **Firefox 88+** - Full support
- **Safari 15+** - Limited WebGL support, may fall back to Canvas 2D
- **Edge 90+** - Full support

## Limitations

- Cross-origin videos must have appropriate CORS headers
- Webcam access requires HTTPS or localhost
- Super resolution is optimized for visual quality, not true detail reconstruction
- Performance depends on GPU capabilities

## License

MIT License
