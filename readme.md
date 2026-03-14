# TTS Pipelining Demo

Benchmarks two approaches to TTS inference and shows how first-audio latency drops from seconds to milliseconds by splitting text into chunks and overlapping inference with playback.

Companion code for the blog post: [Your TTS Is Fast. Your Architecture Isn't.](#)

---

## What it does

**Single-pass** — full text, one inference call. Audio only starts after the entire paragraph is processed.

**Pipelined** — text split into sentence-level chunks. Each chunk is inferred independently. In a real system, chunk N plays while chunk N+1 is being inferred. Here we measure `first_audio_time` to show when audio *could* start.

### Outputs

| File | Description |
|---|---|
| `single_pass.wav` | Full text, one inference call |
| `pipelined_raw.wav` | Chunks stitched without post-processing |
| `pipelined_processed.wav` | Chunks with silence trimmed and punctuation-aware pauses |
| `table1_chunk_split.png` | Visual table of how the text was split |
| `table2_inference_times.png` | Summary latency table |
| `plot1_latency_comparison.png` | First audio latency vs total time |
| `plot2_per_chunk_inference.png` | Per-chunk inference time breakdown |
| `plot3_gantt_timeline.png` | Inference vs playback overlap (Gantt) |
| `plot4_improvement.png` | Latency improvement % |
| `plot5_cumulative_inference.png` | Cumulative inference vs single-pass ceiling |
| `plot6_user_wait_experience.png` | What the user actually experiences |

---

## Setup

### Prerequisites

- Python 3.10+
- [`sox`](https://sox.sourceforge.net/) — used to stitch audio chunks

Install sox:
```bash

# Ubuntu / Debian
sudo apt install sox
```

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python3 blog_inference.py
```

All output files are saved to the current directory.

---

## Requirements

```
kokoro>=0.9.4
soundfile
matplotlib
numpy
```

> Kokoro also requires `espeak-ng` on Linux for phonemization.
> ```bash
> sudo apt install espeak-ng   # Ubuntu / Debian
> brew install espeak           # macOS
> ```

A full `requirements.txt` is included in the repo.

---

## How the splitter works

Naive `.split(".")` breaks on `Dr.`, `3:30 p.m.`, `approx.` and anything else with a period that isn't a sentence boundary. The splitter here uses a regex lookahead to only split where punctuation is followed by actual clause-starting text, and a minimum word count guard to discard short fragments.

It returns `(chunk, delimiter)` pairs so the post-processor knows what punctuation ended each chunk and can apply the right pause duration.

---

## Notes

- The demo runs inference sequentially. In production you'd run the producer (inference) and consumer (playback) on separate threads with a shared queue.
- `first_audio_time` here is the wall-clock time from pipeline start to first chunk ready — this is when playback *would* begin in a real system.
- Pause durations in `PAUSE_CONFIG` are tuned for Kokoro's `af_heart` voice at speed 1. Tune them by ear if you switch models.

---

*Model-agnostic pattern. Kokoro is used here as the example engine — [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on Hugging Face.*
