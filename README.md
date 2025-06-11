# WWDCDigest

Tools for creating digests from Apple WWDC sessions.

## Overview

WWDCDigest is a Python package that provides tools for creating summaries and digests from Apple WWDC session content. It works with the wwdctools package to fetch session data and transcripts, then processes them to create concise digests with video frames extracted at subtitle timestamps.

## Features

- Download video and WebVTT subtitles from WWDC sessions
- Extract video frames at each subtitle timestamp
- Create markdown digests with transcript text and corresponding video frames
- Generate summaries and key points with OpenAI API (optional)
- Translate digests to different languages (with OpenAI)

## Installation

```bash
uv add wwdcdigest
```

## Usage

### Command Line Interface

```bash
# Create a digest from a URL
wwdcdigest digest https://developer.apple.com/videos/play/wwdc2023/10149/

# Create a digest from a localized URL (e.g., Japanese)
wwdcdigest digest https://developer.apple.com/jp/videos/play/wwdc2025/102/

# Specify output directory
wwdcdigest digest https://developer.apple.com/videos/play/wwdc2023/10149/ --output-dir ~/Documents/wwdc_digests

# Use OpenAI to generate summary and key points
wwdcdigest digest https://developer.apple.com/videos/play/wwdc2023/10149/ --openai-key YOUR_API_KEY

# Create digest in a different language (requires OpenAI)
wwdcdigest digest https://developer.apple.com/videos/play/wwdc2023/10149/ --language ja --openai-key YOUR_API_KEY
```

#### Digest Command Options

The `digest` command provides the following options:

- `URL`: (Required) URL of the WWDC session
- `--output-dir`, `-o`: Output directory for generated files. Creates a session subdirectory inside this path.
- `--format`, `-f`: Output format (currently only markdown is supported).
- `--openai-key`: OpenAI API key for generating summary and key points. If not provided, basic digest without AI-generated content will be created.
- `--language`, `-l`: Language code for the digest (e.g., 'en', 'ja', 'zh', 'fr'). Non-English languages require an OpenAI API key.

#### Global CLI Options

These options can be used with any command:

- `--verbose`, `-v`: Enable verbose output for detailed logging.
- `--quiet`, `-q`: Suppress non-error messages.
- `--log-file`: Path to a file where logs should be written.
- `--version`: Show the version and exit.
- `--help`, `-h`: Show help message and exit.

### Python API

```python
import asyncio
from wwdcdigest import create_digest
from wwdcdigest.models import OpenAIConfig

async def main():
    # Create a digest from a URL
    url = "https://developer.apple.com/videos/play/wwdc2023/10149/"
    digest = await create_digest(url)
    print(f"Digest created at: {digest.markdown_path}")

    # Create a digest from a localized URL (e.g., Japanese)
    jp_url = "https://developer.apple.com/jp/videos/play/wwdc2025/102/"
    jp_digest = await create_digest(jp_url)
    print(f"Japanese session digest created at: {jp_digest.markdown_path}")

    # Create a digest with custom output directory
    url = "https://developer.apple.com/videos/play/wwdc2023/10149/"
    digest = await create_digest(url, output_dir="/path/to/output")
    print(f"Frames extracted: {len(digest.segments)}")

    # Generate summary and key points with OpenAI
    url = "https://developer.apple.com/videos/play/wwdc2023/10149/"
    openai_config = OpenAIConfig(api_key="YOUR_API_KEY")
    digest = await create_digest(url, openai_config=openai_config)
    print(f"Summary: {digest.summary}")
    print(f"Key points: {digest.key_points}")

    # Create digest in Japanese (requires OpenAI)
    url = "https://developer.apple.com/videos/play/wwdc2023/10149/"
    openai_config = OpenAIConfig(api_key="YOUR_API_KEY")
    digest = await create_digest(url, openai_config=openai_config, language="ja")
    print(f"Japanese summary: {digest.summary}")

asyncio.run(main())
```

## Output Format

When you run the `digest` command, it creates a structured output in your specified directory with the following components:

1. **Markdown File**: A comprehensive digest containing:

   - Session title and ID
   - Summary (AI-generated if OpenAI key is provided)
   - Key points (AI-generated if OpenAI key is provided)
   - Complete transcript with embedded video frames at each subtitle timestamp

2. **Directory Structure**:
   ```
   output_directory/
   └── wwdc_<session_id>/
       ├── <session_id>_digest.md  # The main digest markdown file
       ├── frames/                 # Directory containing extracted video frames
       │   ├── frame_0001.jpg
       │   ├── frame_0002.jpg
       │   └── ...
       ├── <session_id>.mp4        # The downloaded video file
       └── <session_id>.webvtt     # The downloaded WebVTT subtitle file
   ```

The markdown digest can be viewed in any markdown viewer. The frames are embedded in the markdown file, allowing you to see the video context along with the transcript text.

## Example Output

The generated markdown digest has the following structure:

```markdown
# Session Title

WWDC Session: <session_id>

## Summary

An AI-generated summary of the session content...

## Key Points

- First key insight from the session
- Second important concept covered
- Technical feature or API highlighted
- ...

## Transcript with Video Frames

### 00:00:07.284

Speaker's words at this timestamp...

![Frame at 00:00:07.284](frames/frame_0001.jpg)

---

### 00:00:10.953

Next segment of speech...

![Frame at 00:00:10.953](frames/frame_0002.jpg)

---

... (continues for entire session)
```

This format makes it easy to follow along with the session content, seeing both the transcript and the visual context simultaneously.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/tattn/wwdcdigest.git
cd wwdcdigest

# Create a virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv add --dev .
```

### Running Tests

```bash
uv run --frozen pytest
```

### Checking Code Quality

```bash
uv run --frozen ruff check .
uv run --frozen ruff format .
uv run --frozen pyright
```

## License

MIT
