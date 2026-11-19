<p align="center"><img src="data/dictator.svg" width="128" height="128" alt=""></p>

# Dictator: offline voice dictation for GNOME

Press **Ctrl+Alt+Space**, speak, then press **Enter**. The text is typed where your cursor was: a
messenger, a browser, an editor or a terminal. Speech recognition runs on your computer, with no
account, no API key and no cloud.

- **Private.** Audio stays in memory and never leaves the machine. The speech model runs in a sandbox
  with no network access. Text is typed through GNOME's input method, not pasted through the
  clipboard.
- **Two speech models.** **Small** (the default) is based on
  [tinyasr](https://github.com/dkaraush/tinyasr): 9 languages, very fast, little memory, a 30 to 33 MB
  download per language pack. **Large** is [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3):
  25 European languages, a 2.5 GB download. Turn on the languages you speak, and mix them freely.
- **Works on a plain CPU.** The text is ready about a second after you stop talking, even after
  minutes of dictation: the recording is transcribed in pieces while you speak.
- **Native look.** A small libadwaita-style pill shows your voice level and the icon of the app
  your words will go to. The top-bar icon has a menu with your last texts.

## Install

You need two parts: the GNOME extension, and the **helper program** that records the microphone and
runs the speech model. The extension cannot include the helper (extensions.gnome.org does not allow
executables in extensions), so the helper is installed from this repository:

```sh
git clone https://github.com/kcharukovskyi/dictator.git
cd dictator
./install.sh                  # or ./install.sh --no-extension if you got the extension from extensions.gnome.org
```

Then log out and log in once.

`install.sh` installs into your home folder only and never asks for sudo. It does the following:

1. Checks for `pw-record` and `pw-dump` (PipeWire tools). If one is missing, it prints the install
   command for your distribution.
2. Asks which model you want, Small (the default) or Large, and which languages you speak. It suggests
   your system's language and English. `./install.sh --yes` takes these suggestions.
3. Prepares the model:
   - **Small:** checks for bubblewrap (`bwrap`), the sandbox Small runs in. If it is missing, it
     prints the install command and stops. Then it builds the program from [`engine/`](engine/) on
     your computer, with Rust (`cargo`) if you have it, else in a podman or docker image of Fedora's
     Rust ([`engine/Containerfile`](engine/Containerfile)). The build itself needs no network.
   - **Large:** picks where it runs: the system packages `python3-onnxruntime`, `python3-numpy` and
     `python3-onnx`, sandboxed with bubblewrap, or, if those are missing, a podman or docker image
     that it builds on your computer from signed Fedora packages
     ([`helper/Containerfile`](helper/Containerfile)).
4. Installs the helper into `~/.local/bin/dictate` and the extension.
5. Offers to download the speech model (see below). You can also do this later in the settings
   window.
6. Runs a self-test. It checks that the model works and that its sandbox has no network.

**Requirements:** GNOME 51 on Wayland, PipeWire, a 64-bit CPU (4+ cores recommended). The Small
model also needs bubblewrap: the package `bubblewrap` on Fedora, Debian, Ubuntu and Arch. Free
memory while dictating: about 0.3 GB (Small) or 1.7 GB (Large). Disk: 35 to 38 MB per language pack
(Small) or 1 GB (Large).

## The speech models

Open **microphone menu → Settings → Speech → Speech Model** to pick a model and download it. **Try
Dictation**, in the same section, records a sentence and shows the text right there: it types
nothing anywhere and keeps nothing. In a terminal: `dictate --download small` (the packs for your
languages), `dictate --download int8`, `dictate --models`.

### Small or Large

| | Small (default) | Large |
|---|---|---|
| Model | [tinyasr](https://github.com/dkaraush/tinyasr), with our speed-up | NVIDIA Parakeet TDT 0.6B v3 |
| Languages | 9, in 5 packs: English with Russian and Ukrainian; with Spanish, Italian and Portuguese; with Russian and Uzbek; with Persian; with Khmer | 25 European |
| Download | 30 to 33 MB per pack | 2.5 GB |
| On disk | 35 to 38 MB per pack | 0.9 GB |
| Memory while dictating | about 0.3 GB (one pack) | about 1.7 GB |
| 75 minutes of short clips (FLEURS) | 29 s | 210 s |
| 3 real dictations, 2 minutes of speech | 0.7 s | 5.6 s |

These speed numbers are from one laptop with an Intel Ice Lake CPU (it has AVX-512 VNNI), 8 threads,
and an earlier build of the same engine, timed with tinyasr's test tool with one language on. In
Dictator, each turned-on language that uses the same pack adds one more pass: with Ukrainian, English
and Russian on, Small does three passes. Large runs its encoder once, then decodes once for each
Cyrillic language. The final build is not timed on that laptop yet.

Small picks its CPU code when it starts: AVX-512 VNNI, AVX-VNNI, AVX2 or SSE on Intel and AMD; i8mm,
dotprod or NEON on ARM. AVX2 and SSE ran on a real CPU and under emulation. AVX-512 VNNI ran on the
laptop above, in the earlier build. The AVX-VNNI code has not run yet.

On a server with an AMD EPYC CPU (Zen 2: AVX2, no AVX-512), 8 threads, the original tinyasr against
the final build of Small:

| Test | Original tinyasr | Small |
|---|---|---|
| 19 minutes of short clips (FLEURS, 50 Ukrainian + 50 Russian) | 334 s | 18 s |
| 3 real dictations, 2 minutes of speech | 58 s | 1.7 s |
| Share of wrong words on the same clips, Ukrainian | 7.5% | 7.6% |
| Share of wrong words on the same clips, Russian | 9.5% | 8.8% |

Large was not measured on that server. The ARM code was checked under emulation: 20 of 23 test files
gave the same text as on Intel and AMD, the others differ by a comma or a word. It is not yet timed
on real ARM hardware.

Quality on the laptop above, with the same earlier build: the share of wrong words (WER, lower is
better). Small was measured with the `en-ru-uk` pack.

| Test | Small | Large |
|---|---|---|
| FLEURS, Ukrainian | 6.6% | 7.2% |
| FLEURS, Russian | 7.3% | 6.6% |
| Long audio (about 70 s per file, 20 files in all), Ukrainian | 7.3% | 7.9% |
| Long audio (about 70 s per file, 20 files in all), Russian | 5.9% | 8.8% |

For the other languages we only have tinyasr's own numbers (FLEURS, Small): English 7.6% to 10.3%
depending on the pack, Spanish 5.2%, Italian 4.9%, Portuguese 6.6%, Uzbek 13.5%, Persian 5.9%, Khmer
8.2% (of characters). We have not compared Small and Large on these languages.

Pick **Large** if you need a language that Small does not have. Otherwise Small is the better start.

### Small: where the files come from

- **Source.** The language packs are tinyasr's own files. They are downloaded from
  [its GitHub repository](https://github.com/dkaraush/tinyasr) at a fixed commit. The voice detector
  (Silero VAD, 2 MB) comes with them.
- **Check.** Every file is checked against its SHA-256 before it is used. The hashes are listed in
  [`helper/dictate`](helper/dictate).
- **The program is built on your computer** from the source in [`engine/`](engine/), so there is no
  binary to trust. It runs only in a sandbox: bubblewrap with no network that sees only /usr
  (read-only), the program and the packs. Without bubblewrap it does not start.

### Large: two variants

| Variant | Download | On disk | Notes |
|---|---|---|---|
| **Standard** (8-bit) | 2.5 GB | 0.9 GB | The better of the two. Faster, needs less memory. |
| **Full Precision** (32-bit) | 2.5 GB | 2.5 GB | Slightly more accurate, about half as fast, needs ~3 GB of memory. |

Where the files come from:

- **Source.** The files are NVIDIA's model in the ONNX format, made by the
  [onnx-asr](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) project. They are downloaded
  from Hugging Face at a fixed revision.
- **Check.** Every file is checked against its SHA-256 before it is used. The hashes are listed in
  [`helper/dictate`](helper/dictate).
- **Network.** The download is the only time Dictator uses the network, and it happens only when you
  ask for it.
- **The Standard variant is made on your computer.** Dictator converts the downloaded files to 8-bit
  with ONNX Runtime and then deletes the 32-bit ones. The conversion is reproducible: with ONNX
  Runtime 1.22 the result is byte-for-byte identical to the reference build.
- **Why not a ready-made 8-bit download.** The onnx-asr repository also has 8-bit files, which are
  smaller. On a small test (2.5 minutes of Ukrainian speech, one speaker), their text differed from
  the 32-bit model in 30% of the words. Dictator's own conversion differed in 7%.

The same can be done in a terminal: `dictate --download int8` (or `fp32`), `dictate --remove fp32`.

## Use

| Keys | Action |
|---|---|
| **Ctrl+Alt+Space** | Start recording |
| **Enter** (while recording) | Finish. The text is typed in; the Enter key does not reach the app. Can be another key, Shift+Caps Lock, or the start shortcut again |
| **Caps Lock** (while recording) | Pause / resume. Caps Lock itself is not toggled |
| **Esc** (while recording) | Cancel |

You can change every key in Settings, and changes apply at once. Settings refuses shortcuts that
GNOME already owns (for example, Super+Space switches the input source).

- The text goes to the window where you **started** recording, even if you switched away meanwhile.
- Drag the recording indicator with the mouse to move it; it stays there. A double-click puts it back.
- Long dictation: while you speak, the recording is cut at your pauses into pieces of 10 to 30
  seconds, and each piece is transcribed in the background. Pieces of that length also give better
  text than one long piece, which can drop whole phrases. Stretches with no voice are skipped, so
  the model cannot invent words from noise.
- Some apps don't accept typed text (some X11 and Electron apps). For them, the text waits in the
  top-bar menu, where one click copies it. If you allow it in Settings, Dictator pastes the text
  through the clipboard instead, then puts back what the clipboard held before.
- If the screen locks while you record, the recording is finished and transcribed. Nothing is typed on
  the lock screen; the text waits in the menu.
- Speech settings:
  - Spoken punctuation: say "кома", "comma" or "new line".
  - Removal of hesitation sounds.
  - Word replacements and snippets ("мій імейл = me@example.com").
  - Stop on silence.
  - Choice of microphone.
  - Muting of quieter voices around you.

## Privacy

- Recordings are kept in `$XDG_RUNTIME_DIR` (RAM) only until they are transcribed. A failed recording
  stays there so that you can retry it from the menu. All recordings are deleted after the time set in
  Settings (1 hour by default).
- The menu keeps your last three texts in RAM. You can turn this off.
- Only if you turn on **Keep Audio to Transcribe Again** (Settings, Privacy), their recordings stay in
  RAM too, for the same time, so the menu can transcribe a text again: as it was, or in one of the
  languages you choose there. Off by default; turning it off deletes the kept audio.
- The log (`~/.cache/dictate/dictate.log`) contains timings and numbers, never your text.
- No telemetry.

## Uninstall

```sh
./uninstall.sh          # keeps the models and your settings
./uninstall.sh --all    # removes everything
```

## Project layout

| Path | What |
|---|---|
| `extension/` | The GNOME Shell extension (GJS). `scripts/pack.sh` builds the zip for extensions.gnome.org |
| `helper/dictate` | The helper: recording, model download, transcription (Python, standard library only) |
| `helper/parakeet.py` | The speech model runner (ONNX Runtime + NumPy), always run in the sandbox |
| `helper/Containerfile` | The optional local container image for the Large model |
| `engine/` | The Small model: the tinyasr library (our fork) and its worker `dictate-tinyasr`, in Rust ([`engine/README.md`](engine/README.md)) |
| `docs/REVIEW-NOTES.md` | Notes for extensions.gnome.org reviewers |

## License

Dictator is made by [kcharukovskyi](https://github.com/kcharukovskyi) and [Yegor Isakov](https://isakov.fyi)
([@yegqr](https://github.com/yegqr)). It is licensed GPL-3.0-or-later (see [LICENSE](LICENSE) and
[AUTHORS](AUTHORS)).

The Small model is [tinyasr](https://github.com/dkaraush/tinyasr) by Danil Karaush, GPL-3.0-or-later,
with its language packs. Thanks to Danil Karaush for it. `engine/tinyasr` is our fork of
it, with the same license. The packs were trained mostly on
[YODAS v3](https://huggingface.co/datasets/espnet/yodas3) (ESPnet, CC BY 3.0). The voice detector is
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT).

The Large model is NVIDIA's and licensed CC BY 4.0.

Dictator writes the attribution of each model to `NOTICE.txt` in its folder.

NVIDIA and Parakeet are names of NVIDIA Corporation, used only to identify the model. Dictator is not
affiliated with or endorsed by NVIDIA, and is an independent extension for GNOME.
