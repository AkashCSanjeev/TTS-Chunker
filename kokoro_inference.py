from kokoro import KPipeline
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import subprocess
import time
import re
import os

pipeline = KPipeline(lang_code='a')

TEXT = (
    'Dr. Alan Turing once said "a machine that thinks is a machine that speaks." '
    'Today, voice interfaces are everywhere, from smart home devices to in-car assistants '
    'running on low-end embedded hardware. Consider a typical assistant response: it must '
    'greet the user, confirm the task, and deliver results. For example, "Your meeting with '
    'Dr. Smith is at 3:30 p.m. The drive will take approx. 24.5 km, or about 35 min. via '
    'the M4 motorway." Can the system deliver this fast enough to feel natural? That is the '
    'real question. On embedded hardware, think Raspberry Pi 4, or an i.MX 8M, every '
    'millisecond counts. A delay of even 3 sec. feels broken to a user standing in front '
    'of the device.'
)

# ---------------------------------------------------------------------------
# Splitter
# Splits on punctuation, skips fragments shorter than min_words to avoid
# breaking on abbreviations like Dr. or p.m.
# Returns (chunk, delimiter) pairs so we know what caused each split.
# ---------------------------------------------------------------------------
def split_by_punctuation(text, min_words=3):
    pattern  = re.compile(r'[.,:?"()](?!(?:\d|\s*$))')
    results  = []
    last_pos = 0

    for match in pattern.finditer(text):
        candidate = text[last_pos:match.start()].strip()

        if len(candidate.split()) < min_words:
            continue

        delimiter = match.group()

        # closing quote followed by . or , — use the trailing punctuation
        if delimiter == '"' and match.end() < len(text):
            next_char = text[match.end()]
            if next_char in ('.', ','):
                delimiter = next_char
                results.append((candidate, delimiter))
                last_pos = match.end() + 1
                continue

        results.append((candidate, delimiter))
        last_pos = match.end()

    trailing = text[last_pos:].strip()
    if trailing:
        results.append((trailing, ""))

    return results


# ---------------------------------------------------------------------------
# Silence trim + punctuation-aware padding
# TTS models pad silence on every clip. Stitching 19 clips means 19 stacked
# paddings. Strip it, then add back deliberate pauses based on the delimiter.
# ---------------------------------------------------------------------------
SAMPLE_RATE       = 24000
SILENCE_THRESHOLD = 0.0025

PAUSE_CONFIG = {
    ".":  {"lead": 0.15, "trail": 0.15},
    ",":  {"lead": 0.15, "trail": 0.05},
    "?":  {"lead": 0.15, "trail": 0.15},
    "!":  {"lead": 0.15, "trail": 0.15},
    ":":  {"lead": 0.15, "trail": 0.05},
    "":   {"lead": 0.10, "trail": 0.10},
}

def trim_silence(audio, threshold=SILENCE_THRESHOLD, min_non_silence=0.01):
    min_samples = int(min_non_silence * SAMPLE_RATE)
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0:
        return audio
    start = max(0, above[0] - min_samples)
    end   = min(len(audio), above[-1] + min_samples)
    return audio[start:end]

def apply_pause(audio, delimiter):
    cfg       = PAUSE_CONFIG.get(delimiter, PAUSE_CONFIG[""])
    audio     = trim_silence(audio)
    lead_pad  = np.zeros(int(cfg["lead"]  * SAMPLE_RATE), dtype=audio.dtype)
    trail_pad = np.zeros(int(cfg["trail"] * SAMPLE_RATE), dtype=audio.dtype)
    return np.concatenate([lead_pad, audio, trail_pad])


# ---------------------------------------------------------------------------
# 1. Single-pass — full text, one inference call, no splitting
# ---------------------------------------------------------------------------
print("=" * 50)
print("SINGLE-PASS INFERENCE")
print("=" * 50)

start = time.time()

# split_pattern=None forces Kokoro to treat the entire TEXT as one unit
gen      = pipeline(TEXT, voice='af_heart', speed=1, split_pattern=None)
parts    = [audio.numpy() if hasattr(audio, 'numpy') else np.array(audio) for _, _, audio in gen]
audio_sp = np.concatenate(parts) if len(parts) > 1 else parts[0]

single_pass_time = (time.time() - start) * 1000
sf.write("single_pass.wav", audio_sp, SAMPLE_RATE)

print(f"Inference time: {single_pass_time:.1f} ms")
print("Saved: single_pass.wav")


# ---------------------------------------------------------------------------
# 2. Pipelined — text split into chunks, each inferred independently
# In production, chunk N plays while chunk N+1 is being inferred.
# Here we measure first_audio_time = time to first chunk ready.
# ---------------------------------------------------------------------------
print()
print("=" * 50)
print("PIPELINED INFERENCE")
print("=" * 50)

chunks = split_by_punctuation(TEXT)
print(f"Total chunks: {len(chunks)}\n")
for i, (chunk, delimiter) in enumerate(chunks):
    delim_display = repr(delimiter) if delimiter else "''"
    print(f"  [{i+1:02d}] delim={delim_display:<6} | {chunk}")
print()

chunk_times      = []
first_audio_time = None
pipeline_start   = time.time()

for i, (chunk, delimiter) in enumerate(chunks):
    chunk_start = time.time()

    gen        = pipeline(chunk, voice='af_heart', speed=1, split_pattern=None)
    parts      = [audio.numpy() if hasattr(audio, 'numpy') else np.array(audio) for _, _, audio in gen]
    raw_audio  = np.concatenate(parts) if len(parts) > 1 else parts[0]

    sf.write(f"chunk_{i+1}_raw.wav", raw_audio, SAMPLE_RATE)
    sf.write(f"chunk_{i+1}_processed.wav", apply_pause(raw_audio, delimiter), SAMPLE_RATE)

    elapsed = (time.time() - chunk_start) * 1000
    chunk_times.append(elapsed)

    if first_audio_time is None:
        first_audio_time = (time.time() - pipeline_start) * 1000

    cfg = PAUSE_CONFIG.get(delimiter, PAUSE_CONFIG[""])
    print(f"Chunk {i+1:02d} [{elapsed:.0f}ms] lead={cfg['lead']}s trail={cfg['trail']}s | {chunk[:60]}")

total_pipeline_time = (time.time() - pipeline_start) * 1000
avg_chunk_time      = sum(chunk_times) / len(chunk_times)

raw_files       = " ".join([f"chunk_{i+1}_raw.wav"       for i in range(len(chunks))])
processed_files = " ".join([f"chunk_{i+1}_processed.wav" for i in range(len(chunks))])
os.system(f"sox {raw_files} pipelined_raw.wav")
os.system(f"sox {processed_files} pipelined_processed.wav")
print("\nSaved: pipelined_raw.wav, pipelined_processed.wav")


# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"{'Approach':<30} {'First Audio':>15} {'Avg/Chunk':>12} {'Total':>12}")
print("-" * 70)
print(f"{'Single-pass':<30} {'~'+str(round(single_pass_time))+'ms':>15} {'—':>12} {'~'+str(round(single_pass_time))+'ms':>12}")
print(f"{'Pipelined':<30} {'~'+str(round(first_audio_time))+'ms':>15} {'~'+str(round(avg_chunk_time))+'ms':>12} {'~'+str(round(total_pipeline_time))+'ms':>12}")


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

DARK_BG   = "#ffffff"
CARD_BG   = "#f8f9fb"
BORDER    = "#d0d5e0"
BLUE      = "#2563eb"
BLUE2     = "#1d4ed8"
GREEN     = "#16a34a"
GREEN2    = "#15803d"
ORANGE    = "#d97706"
RED       = "#dc2626"
PURPLE    = "#7c3aed"
TEXT_COL  = "#1e293b"
MUTED     = "#94a3b8"
GRID_COL  = "#e2e8f0"
HEADER_BG = "#dbeafe"
ROW_A     = "#ffffff"
ROW_B     = "#f1f5f9"

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize':10,
    'ytick.labelsize':10,
    'legend.fontsize':10,
    'figure.dpi':     150,
})

def style_ax(ax):
    ax.set_facecolor(CARD_BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)
    ax.tick_params(colors=TEXT_COL, length=4)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    ax.grid(axis='y', color=GRID_COL, linewidth=0.8, zorder=0)

def save_fig(fig, filename):
    fig.patch.set_facecolor(DARK_BG)
    fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved: {filename}")


# Table 1: chunk split
n_chunks   = len(chunks)
ROW_H_T    = 0.032
fig_height = max(3.5, n_chunks * ROW_H_T * 20 + 1.8)
fig, ax    = plt.subplots(figsize=(14, fig_height))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis('off')
ax.set_title("Chunk Split Table — Sentence & Delimiter",
             color=TEXT_COL, fontsize=13, fontweight='bold', pad=12)

col_labels = ["#", "Delimiter", "Chunk Text"]
x_pos      = [0.01, 0.06, 0.15]
table_top  = 0.92

ax.add_patch(plt.Rectangle((0, table_top), 1.0, ROW_H_T,
             transform=ax.transAxes, color=HEADER_BG, clip_on=False, zorder=2))
for label, xp in zip(col_labels, x_pos):
    ax.text(xp, table_top + ROW_H_T / 2, label,
            transform=ax.transAxes, color=ORANGE,
            fontsize=11, fontweight='bold', va='center')

for i, (chunk, delimiter) in enumerate(chunks):
    y = table_top - (i + 1) * ROW_H_T
    ax.add_patch(plt.Rectangle((0, y), 1.0, ROW_H_T,
                 transform=ax.transAxes,
                 color=ROW_A if i % 2 == 0 else ROW_B,
                 clip_on=False, zorder=1))
    delim_display = repr(delimiter) if delimiter else "none"
    delim_color   = (GREEN if delimiter in ('.', '?', '!') else
                     BLUE  if delimiter in (',', ':') else MUTED)
    ax.text(x_pos[0], y + ROW_H_T / 2, str(i + 1),
            transform=ax.transAxes, color=MUTED, fontsize=10, va='center')
    ax.text(x_pos[1], y + ROW_H_T / 2, delim_display,
            transform=ax.transAxes, color=delim_color,
            fontsize=10, fontweight='bold', va='center', fontfamily='monospace')
    preview = chunk if len(chunk) <= 100 else chunk[:97] + "..."
    ax.text(x_pos[2], y + ROW_H_T / 2, preview,
            transform=ax.transAxes, color=TEXT_COL, fontsize=10, va='center')
    ax.plot([0, 1], [y, y], color=BORDER,
            linewidth=0.5, transform=ax.transAxes, clip_on=False)

ax.plot([0, 1], [table_top - n_chunks * ROW_H_T] * 2,
        color=BORDER, linewidth=0.8, transform=ax.transAxes, clip_on=False)
ax.legend(handles=[
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=GREEN, markersize=9, label="Sentence end (. ? !)"),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=BLUE,  markersize=9, label="Pause (, :)"),
    plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=MUTED, markersize=9, label="No delimiter"),
], loc='lower right', facecolor=CARD_BG, labelcolor=TEXT_COL,
   framealpha=0.9, edgecolor=BORDER, fontsize=10)
plt.tight_layout(pad=1.2)
save_fig(fig, "table1_chunk_split.png")


# Table 2: inference time summary
table_data = [
    ("Single-pass (full wav)",  f"{single_pass_time:.1f} ms",   RED),
    ("Pipelined — First chunk", f"{first_audio_time:.1f} ms",   GREEN),
    ("Pipelined — Avg/chunk",   f"{avg_chunk_time:.1f} ms",     ORANGE),
    ("Pipelined — Total",       f"{total_pipeline_time:.1f} ms", BLUE),
]

fig, ax = plt.subplots(figsize=(7, 3.0))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.axis('off')
ax.set_title("Inference Time Summary", color=TEXT_COL,
             fontsize=13, fontweight='bold', pad=12)

x_pos2     = [0.03, 0.72]
table_top2 = 0.82
rh2        = 0.13

ax.add_patch(plt.Rectangle((0, table_top2), 1.0, rh2,
             transform=ax.transAxes, color=HEADER_BG, clip_on=False, zorder=2))
for label, xp in zip(["Approach", "Time (ms)"], x_pos2):
    ax.text(xp, table_top2 + rh2 / 2, label,
            transform=ax.transAxes, color=ORANGE,
            fontsize=11, fontweight='bold', va='center')

for i, (label, time_val, vc) in enumerate(table_data):
    y = table_top2 - (i + 1) * rh2
    ax.add_patch(plt.Rectangle((0, y), 1.0, rh2,
                 transform=ax.transAxes,
                 color=ROW_A if i % 2 == 0 else ROW_B,
                 clip_on=False, zorder=1))
    ax.text(x_pos2[0], y + rh2 / 2, label,
            transform=ax.transAxes, color=TEXT_COL,
            fontsize=10.5, fontweight='bold', va='center')
    ax.text(x_pos2[1], y + rh2 / 2, time_val,
            transform=ax.transAxes, color=vc,
            fontsize=10.5, fontweight='bold', va='center')
    ax.plot([0, 1], [y, y], color=BORDER,
            linewidth=0.5, transform=ax.transAxes, clip_on=False)

ax.plot([0, 1], [table_top2 - len(table_data) * rh2] * 2,
        color=BORDER, linewidth=0.8, transform=ax.transAxes, clip_on=False)
plt.tight_layout(pad=1.2)
save_fig(fig, "table2_inference_times.png")


# Plot 1: first audio latency vs total time
fig, ax = plt.subplots(figsize=(9, 6))
style_ax(ax)

categories = ["Single-pass", "Pipelined"]
x          = np.arange(len(categories))
width      = 0.32

b1 = ax.bar(x - width/2, [single_pass_time, total_pipeline_time], width,
            label="Total Inference Time", color=BLUE, zorder=3, linewidth=0)
b2 = ax.bar(x + width/2, [single_pass_time, first_audio_time],    width,
            label="First Audio Latency", color=GREEN, zorder=3, linewidth=0)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + single_pass_time * 0.015,
            f"{h:.0f} ms", ha='center', va='bottom',
            color=TEXT_COL, fontsize=10, fontweight='bold')

ax.annotate(
    f"↓ {single_pass_time - first_audio_time:.0f} ms faster\nto first audio",
    xy=(1 + width/2, first_audio_time),
    xytext=(1 + width/2 + 0.35, first_audio_time + single_pass_time * 0.15),
    fontsize=9, color=ORANGE, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5),
)

ax.set_xticks(x)
ax.set_xticklabels(categories, color=TEXT_COL)
ax.set_ylabel("Time (ms)", color=TEXT_COL)
ax.set_ylim(0, max(single_pass_time, total_pipeline_time) * 1.35)
ax.set_title("First Audio Latency vs Total Inference Time", pad=14)
ax.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, framealpha=0.0, edgecolor=BORDER)
plt.tight_layout(pad=2.5)
save_fig(fig, "plot1_latency_comparison.png")


# Plot 2: per-chunk inference time
fig, ax = plt.subplots(figsize=(max(10, len(chunk_times) * 0.7), 6))
style_ax(ax)

chunk_labels = [f"C{i+1}" for i in range(len(chunk_times))]
bar_colors   = [GREEN if t <= avg_chunk_time else ORANGE for t in chunk_times]
bars = ax.bar(chunk_labels, chunk_times, color=bar_colors, zorder=3, linewidth=0, width=0.6)

ax.axhline(avg_chunk_time,   color=RED,    linestyle='--', linewidth=1.8, label=f"Avg: {avg_chunk_time:.0f} ms",          zorder=4)
ax.axhline(first_audio_time, color=PURPLE, linestyle=':',  linewidth=1.8, label=f"First audio: {first_audio_time:.0f} ms", zorder=4)

for bar, t in zip(bars, chunk_times):
    ax.text(bar.get_x() + bar.get_width()/2, t + max(chunk_times) * 0.015,
            f"{t:.0f}", ha='center', va='bottom', color=TEXT_COL, fontsize=8, fontweight='bold')

ax.legend(handles=[
    plt.Rectangle((0,0),1,1, fc=GREEN,  label="Below avg"),
    plt.Rectangle((0,0),1,1, fc=ORANGE, label="Above avg"),
    plt.Line2D([0],[0], color=RED,    linestyle='--', lw=1.8, label=f"Avg: {avg_chunk_time:.0f} ms"),
    plt.Line2D([0],[0], color=PURPLE, linestyle=':',  lw=1.8, label=f"First audio: {first_audio_time:.0f} ms"),
], facecolor=CARD_BG, labelcolor=TEXT_COL, framealpha=0.0, edgecolor=BORDER)

ax.set_xlabel("Chunk", color=TEXT_COL)
ax.set_ylabel("Inference Time (ms)", color=TEXT_COL)
ax.set_ylim(0, max(chunk_times) * 1.3)
ax.set_title("Per-Chunk Inference Time Breakdown", pad=14)
for lbl in ax.get_xticklabels():
    lbl.set_color(TEXT_COL)
plt.tight_layout(pad=2.5)
save_fig(fig, "plot2_per_chunk_inference.png")


# Plot 3: Gantt — inference vs playback overlap
# Playback durations come from actual processed wav files via soxi, not inference time.
fig, ax = plt.subplots(figsize=(13, 5))
style_ax(ax)
ax.grid(axis='x', color=GRID_COL, linewidth=0.8, zorder=0)
ax.grid(axis='y', visible=False)

play_durs_ms = []
for i in range(len(chunks)):
    result = subprocess.run(["soxi", "-D", f"chunk_{i+1}_processed.wav"],
                            capture_output=True, text=True)
    play_durs_ms.append(float(result.stdout.strip()) * 1000)

infer_starts = []
t_acc = 0
for ct in chunk_times:
    infer_starts.append(t_acc)
    t_acc += ct

play_starts, play_durs = [], []
play_t = infer_starts[0] + chunk_times[0]
for i, pd in enumerate(play_durs_ms):
    play_starts.append(play_t)
    play_durs.append(pd)
    if i + 1 < len(chunks):
        play_t = max(play_t + pd, infer_starts[i+1] + chunk_times[i+1])
    else:
        play_t += pd

ROW_H   = 0.38
Y_INFER = 1.9
Y_PLAY  = 0.8

total_playback_end = play_starts[-1] + play_durs[-1]
x_infer_end        = sum(chunk_times)
x_max              = total_playback_end * 1.06

ax.axvspan(x_infer_end, total_playback_end, alpha=0.07, color=GREEN, zorder=0, label="Playback-only zone")
ax.axvline(x_infer_end, color=BLUE, linestyle=':', linewidth=1.5, zorder=4, alpha=0.7)
ax.text(x_infer_end + x_max * 0.005, 2.25,
        f"Inference done\n{x_infer_end:.0f} ms",
        color=BLUE, fontsize=8, va='top', alpha=0.85)

for i, (s, ct) in enumerate(zip(infer_starts, chunk_times)):
    ax.barh(Y_INFER, ct, height=ROW_H, left=s,
            color=BLUE if i % 2 == 0 else BLUE2, alpha=0.9, zorder=3, linewidth=0)
    if ct > x_infer_end * 0.03:
        ax.text(s + ct / 2, Y_INFER, f"C{i+1}",
                ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')

for i, (s, d) in enumerate(zip(play_starts, play_durs)):
    ax.barh(Y_PLAY, d, height=ROW_H, left=s,
            color=GREEN if i % 2 == 0 else GREEN2, alpha=0.9, zorder=3, linewidth=0)
    if d > x_max * 0.02:
        ax.text(s + d / 2, Y_PLAY, f"C{i+1}",
                ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')

for i in range(min(3, len(infer_starts) - 1)):
    overlap_start = play_starts[i]
    overlap_end   = min(play_starts[i] + play_durs[i], infer_starts[i+1] + chunk_times[i+1])
    if overlap_end > overlap_start:
        mid = (overlap_start + overlap_end) / 2
        ax.annotate("", xy=(mid, Y_PLAY + ROW_H/2 + 0.05),
                    xytext=(mid, Y_INFER - ROW_H/2 - 0.05),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2))

ax.axvline(first_audio_time, color=ORANGE, linestyle='--', linewidth=2, zorder=5)
ax.text(first_audio_time + x_max * 0.008, Y_PLAY - 0.38,
        f"First audio\n{first_audio_time:.0f} ms",
        color=ORANGE, fontsize=8.5, fontweight='bold', va='top')

ax.set_yticks([Y_PLAY, Y_INFER])
ax.set_yticklabels(["▶  Playback (real-time)", "⚙  Inference (chunked)"],
                   color=TEXT_COL, fontsize=10)
ax.set_xlabel("Time (ms)", color=TEXT_COL)
ax.set_xlim(0, x_max)
ax.set_ylim(0.3, 2.6)
ax.set_title("Pipeline Timeline — Inference vs Playback Overlap (Gantt)", pad=14)
ax.legend(handles=[
    mpatches.Patch(color=BLUE,  alpha=0.9,  label="Chunk inference"),
    mpatches.Patch(color=GREEN, alpha=0.9,  label="Chunk playback"),
    mpatches.Patch(color=GREEN, alpha=0.12, label="Playback-only zone"),
], loc='upper right', facecolor=CARD_BG, labelcolor=TEXT_COL,
   framealpha=0.9, edgecolor=BORDER, fontsize=9)
plt.tight_layout(pad=2.5)
save_fig(fig, "plot3_gantt_timeline.png")


# Plot 4: latency improvement %
fig, ax = plt.subplots(figsize=(8, 5))
style_ax(ax)
ax.grid(axis='x', color=GRID_COL, linewidth=0.8, zorder=0)
ax.grid(axis='y', visible=False)

metrics     = ["Total Time", "First Audio\nLatency"]
single_vals = [single_pass_time, single_pass_time]
pipe_vals   = [total_pipeline_time, first_audio_time]
savings_pct = [(s - p) / s * 100 for s, p in zip(single_vals, pipe_vals)]
y, h        = np.arange(len(metrics)), 0.28

ax.barh(y + h/2, single_vals, h, label="Single-pass", color=RED,   alpha=0.85, zorder=3, linewidth=0)
ax.barh(y - h/2, pipe_vals,   h, label="Pipelined",   color=GREEN, alpha=0.85, zorder=3, linewidth=0)

max_val = max(single_vals)
for i, (sv, pv, pct) in enumerate(zip(single_vals, pipe_vals, savings_pct)):
    ax.text(sv + max_val * 0.01, i + h/2, f"{sv:.0f} ms", va='center', color=TEXT_COL, fontsize=9)
    ax.text(pv + max_val * 0.01, i - h/2, f"{pv:.0f} ms", va='center', color=TEXT_COL, fontsize=9)
    ax.text(max_val * 1.18, i, f"↓ {pct:.0f}%", va='center', ha='center',
            color=ORANGE, fontsize=12, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(metrics, color=TEXT_COL)
ax.set_xlabel("Time (ms)", color=TEXT_COL)
ax.set_xlim(0, max_val * 1.35)
ax.set_title("Latency Improvement: Single-pass vs Pipelined", pad=14)
ax.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, framealpha=0.0, edgecolor=BORDER)
plt.tight_layout(pad=2.5)
save_fig(fig, "plot4_improvement.png")


# Plot 5: cumulative inference time
fig, ax = plt.subplots(figsize=(10, 6))
style_ax(ax)

cumulative = np.cumsum(chunk_times)
chunk_idxs = np.arange(1, len(chunk_times) + 1)

ax.plot(chunk_idxs, cumulative, color=GREEN, linewidth=2.5,
        marker='o', markersize=6, zorder=4, label="Pipelined cumulative")
ax.axhline(single_pass_time, color=RED, linestyle='--', linewidth=2,
           label=f"Single-pass total: {single_pass_time:.0f} ms", zorder=3)
ax.fill_between(chunk_idxs, cumulative, single_pass_time,
                where=(cumulative < single_pass_time),
                color=GREEN, alpha=0.08, zorder=2)

ax.scatter([1], [chunk_times[0]], color=ORANGE, s=80, zorder=5)
ax.annotate(f"First audio\n{chunk_times[0]:.0f} ms",
            xy=(1, chunk_times[0]),
            xytext=(1.6, chunk_times[0] + single_pass_time * 0.08),
            fontsize=9, color=ORANGE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5))

for xi, yi in zip(chunk_idxs, cumulative):
    ax.text(xi, yi + single_pass_time * 0.015, f"{yi:.0f}",
            ha='center', fontsize=8, color=TEXT_COL)

ax.set_xticks(chunk_idxs)
ax.set_xticklabels([f"C{i}" for i in chunk_idxs], color=TEXT_COL)
ax.set_xlabel("Chunk", color=TEXT_COL)
ax.set_ylabel("Cumulative Time (ms)", color=TEXT_COL)
ax.set_ylim(0, max(single_pass_time, cumulative[-1]) * 1.25)
ax.set_title("Cumulative Inference Time — Pipelined vs Single-pass Ceiling", pad=14)
ax.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, framealpha=0.0, edgecolor=BORDER)
plt.tight_layout(pad=2.5)
save_fig(fig, "plot5_cumulative_inference.png")


# Plot 6: user wait experience
fig, ax = plt.subplots(figsize=(11, 5))
style_ax(ax)
ax.grid(axis='x', color=GRID_COL, linewidth=0.8, zorder=0)
ax.grid(axis='y', visible=False)

timeline_end = max(single_pass_time, play_starts[-1] + play_durs[-1]) * 1.08
BAR_H    = 0.4
Y_SINGLE = 2.0
Y_PIPE   = 1.0

ax.barh(Y_SINGLE, single_pass_time,                   height=BAR_H, left=0,                color=RED,   alpha=0.75, zorder=3, linewidth=0, label="Silence (waiting)")
ax.barh(Y_SINGLE, timeline_end - single_pass_time,    height=BAR_H, left=single_pass_time, color=GREEN, alpha=0.75, zorder=3, linewidth=0, label="Audio playing")
ax.barh(Y_PIPE,   first_audio_time,                   height=BAR_H, left=0,                color=RED,   alpha=0.75, zorder=3, linewidth=0)
ax.barh(Y_PIPE,   timeline_end - first_audio_time,    height=BAR_H, left=first_audio_time, color=GREEN, alpha=0.75, zorder=3, linewidth=0)

ax.text(single_pass_time / 2, Y_SINGLE,
        f"Wait: {single_pass_time:.0f} ms",
        ha='center', va='center', color='white', fontsize=9.5, fontweight='bold')

min_label_width = timeline_end * 0.08
if first_audio_time > min_label_width:
    ax.text(first_audio_time / 2, Y_PIPE,
            f"Wait: {first_audio_time:.0f} ms",
            ha='center', va='center', color='white', fontsize=9.5, fontweight='bold')
else:
    ax.text(first_audio_time + timeline_end * 0.01, Y_PIPE + BAR_H / 2 + 0.07,
            f"Wait: {first_audio_time:.0f} ms",
            ha='left', va='bottom', color=RED, fontsize=9, fontweight='bold')

ax.text(first_audio_time + (timeline_end - first_audio_time) / 2, Y_PIPE,
        f"Audio: {(timeline_end - first_audio_time):.0f} ms",
        ha='center', va='center', color='white', fontsize=9.5, fontweight='bold')

ax.axvline(first_audio_time, color=ORANGE, linestyle='--', linewidth=2, zorder=5)
ax.text(first_audio_time + timeline_end * 0.005, Y_PIPE - BAR_H / 2 - 0.12,
        f"First audio\n{first_audio_time:.0f} ms",
        ha='left', va='top', color=ORANGE, fontsize=8.5, fontweight='bold')

saving_pct = (single_pass_time - first_audio_time) / single_pass_time * 100
ax.annotate(
    f"↓ {saving_pct:.0f}% faster\nto first audio",
    xy=(first_audio_time, (Y_SINGLE + Y_PIPE) / 2),
    xytext=(first_audio_time + timeline_end * 0.12, (Y_SINGLE + Y_PIPE) / 2),
    fontsize=9, color=ORANGE, fontweight='bold', va='center',
    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5),
)

ax.set_yticks([Y_PIPE, Y_SINGLE])
ax.set_yticklabels(["Pipelined", "Single-pass"], color=TEXT_COL, fontsize=11)
ax.set_xlabel("Time (ms)", color=TEXT_COL)
ax.set_xlim(0, timeline_end)
ax.set_ylim(0.55, 2.6)
ax.set_title("User Experience: How Long Before Audio Starts?", pad=14)
ax.legend(facecolor=CARD_BG, labelcolor=TEXT_COL, framealpha=0.0,
          edgecolor=BORDER, loc='upper right')
plt.tight_layout(pad=2.5)
save_fig(fig, "plot6_user_wait_experience.png")

print("\nDone.")