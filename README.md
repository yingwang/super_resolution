# Video Compression Comparison

A web application that demonstrates real video compression using WebRTC encoding/decoding. Compare original uncompressed video with compressed video side-by-side to see the effects of different codecs and bitrates.

## Features

- **Real Video Compression**
  - Uses WebRTC's actual video codecs (VP8, VP9, H.264)
  - Video goes through real encoding/decoding pipeline
  - Same compression used in video calls and streaming

- **Side-by-Side Comparison**
  - Original (uncompressed) video on the left
  - Compressed video on the right
  - See compression artifacts in real-time

- **Multiple Video Sources**
  - Sample videos from Google's public video bucket
  - Custom URL input (MP4, WebM)
  - Webcam capture via WebRTC getUserMedia
  - Screen capture via getDisplayMedia

- **Configurable Compression**
  - **Codec Selection**: VP8, VP9, H.264
  - **Bitrate Control**: 100 kbps to 8000 kbps
  - Fine-tune slider for precise bitrate adjustment

- **Real-time Statistics**
  - Actual bitrate vs target bitrate
  - Resolution and FPS
  - Packets lost and frames dropped
  - Data sent tracking

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
   - Choose from sample videos, enter a custom URL, or use webcam/screen share

2. **Configure Compression**
   - Select codec (VP8, VP9, H.264)
   - Set bitrate (lower = more compression artifacts)
   - Use presets or fine-tune with slider

3. **Compare Quality**
   - Click Play to start both videos
   - Watch the side-by-side comparison
   - Lower bitrates show more visible compression artifacts

### Tips for Seeing Compression Artifacts

- **100-250 kbps**: Very obvious blocking, blur, and color banding
- **500 kbps**: Noticeable artifacts during motion
- **1000+ kbps**: Subtle artifacts, good quality for most content
- **4000+ kbps**: Near-transparent quality

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space / K | Play/Pause |
| F | Toggle Fullscreen |
| ← / → | Seek -5s / +5s |

## Architecture

```
super_resolution/
├── index.html              # Main HTML page
├── styles.css              # Application styles
├── js/
│   ├── app.js              # Main application logic
│   ├── compression-loopback.js  # WebRTC compression loopback
│   ├── enhancer.js         # Video enhancement (optional)
│   └── video-processor.js  # Video streaming utilities
├── package.json
└── README.md
```

### Key Components

- **CompressionLoopback** (`js/compression-loopback.js`)
  - Creates WebRTC peer connections for local loopback
  - Actual video encoding using browser's codecs
  - Statistics collection from WebRTC stats API
  - Configurable codec and bitrate

- **App** (`js/app.js`)
  - UI event handling
  - Component orchestration
  - Playback controls

## How It Works

The application creates a local WebRTC loopback:

1. **Source video** → **RTCPeerConnection (Sender)** → Encode with VP8/VP9/H264
2. Encoded video → **RTCPeerConnection (Receiver)** → Decode
3. Decoded video → **Display side-by-side with original**

This is the same encoding/decoding process used in:
- Video conferencing (Zoom, Meet, Teams)
- WebRTC streaming
- Real-time video applications

## Technical Details

### WebRTC Compression Pipeline

1. Video source captured as MediaStream
2. MediaStream added to sender RTCPeerConnection
3. Encoding parameters set (bitrate, codec preference)
4. SDP modified to prefer specific codec
5. Encoded frames sent to receiver via local ICE
6. Receiver decodes and outputs to video element

### Supported Codecs

| Codec | Notes |
|-------|-------|
| VP8 | Widely supported, good compression |
| VP9 | Better quality at lower bitrates |
| H.264 | Hardware acceleration common |
| AV1 | Best quality (if browser supports) |

### Bitrate Impact

Lower bitrates mean:
- Smaller data size
- More compression artifacts
- Blockiness, especially in motion
- Color banding in gradients
- Loss of fine detail

## Browser Compatibility

- **Chrome 90+** (recommended) - Full VP8/VP9/H264 support
- **Firefox 88+** - Full support
- **Safari 15+** - H.264 support, limited VP9
- **Edge 90+** - Full support

## Limitations

- Cross-origin videos must have appropriate CORS headers
- Webcam access requires HTTPS or localhost
- Actual compression quality depends on browser's encoder implementation
- Local loopback has no network latency/jitter simulation

## License

MIT License
