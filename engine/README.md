# engine: the Small speech model

The code of the Small model, in Rust. `install.sh` builds it on your computer. The Large model
(NVIDIA Parakeet) does not use it.

| Path | What |
|---|---|
| `tinyasr/` | The speech recognition library: our fork of [tinyasr](https://github.com/dkaraush/tinyasr) by Danil Karaush |
| `worker/` | `dictate-tinyasr`, the program that the helper (`helper/dictate`) starts, only inside bubblewrap. It loads the language packs once, then turns audio into text, one request at a time on stdin and stdout, like `helper/parakeet.py --worker` |
| `Containerfile` | The build image that `install.sh` uses when your computer has no Rust |

## Where it comes from

`tinyasr/` is tinyasr at commit
[`1e71924`](https://github.com/dkaraush/tinyasr/tree/1e7192467cfecb66abdd5fd3a2928b06127c7e3f), with our
changes on top:

- **Speed.** Faster CPU code for the model, picked when the program starts, so one build uses the
  best code the CPU has (x86: AVX-512 VNNI, AVX-VNNI, AVX2, SSE; ARM: i8mm, dotprod, NEON). The pieces
  of a long recording are decoded in parallel, and the beam search does less work. On a server with
  an AMD EPYC Zen 2 CPU (AVX2), the final build is 18 to 35 times faster than the original, with
  about the same share of wrong words. On a laptop with an Intel Ice Lake CPU (AVX-512 VNNI), an
  earlier build, timed with tinyasr's test tool, was about 10 times faster; the final build is not
  timed there yet.
- **Quiet speech.** The original could drop whole sentences when the speaker was quiet, because the
  voice detector took them for silence. Ours evens out the volume first.
- **Piece starts.** The helper sends a long dictation in pieces. Started cold in the middle of
  speech, the model could skip the first words of a piece, often a whole sentence, so the end of a
  dictation could go missing. Now it also hears the 4 seconds before each piece, without writing
  them again. On three real dictations (2 minutes), the words lost this way went from 12 to 3.
- **Scores.** Each text comes with a score, so the helper can choose between languages.

Where the CPU code has run so far:

- x86 AVX2 and SSE: on a real AMD CPU and under emulation (qemu).
- x86 AVX-512 VNNI: on the Ice Lake laptop, in the earlier build.
- x86 AVX-VNNI: not run yet.
- ARM (NEON, dotprod, i8mm): under emulation only. 20 of 23 test files gave the same text as on x86;
  the others differ by a comma or a word.
- Windows: a mingw build ran under wine. macOS: it compiles, but has not run.

The Dictator repository holds the code only. The language packs are downloaded by
`dictate --download small` from tinyasr's GitHub repository at the same commit, and each file is
checked against its SHA-256 (listed in `helper/dictate`):

| Pack | Languages | Download | On disk |
|---|---|---|---|
| `en-ru-uk` | English, Russian, Ukrainian | 33 MB | 38 MB |
| `en-es-it-pt` | English, Spanish, Italian, Portuguese | 32 MB | 37 MB |
| `en-ru-uz` | English, Russian, Uzbek | 31 MB | 36 MB |
| `en-fa` | English, Persian | 30 MB | 35 MB |
| `en-km` | English, Khmer | 30 MB | 35 MB |
| `silero_vad.onnx` | the voice detector, with any pack | 2 MB | 2 MB |

## License

- tinyasr is by Danil Karaush and licensed GPL-3.0-or-later ([`tinyasr/LICENSE`](tinyasr/LICENSE)).
  Our changes and `worker/` have the same license.
- The language packs are part of tinyasr and have its license, GPL-3.0-or-later. They were trained
  mostly on [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) (ESPnet,
  [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)), a corpus of Creative Commons YouTube
  audio; the dataset card has the attribution per source.
- The voice detector is [Silero VAD](https://github.com/snakers4/silero-vad) (MIT).

The helper writes this attribution to `NOTICE.txt` in the pack folder (`~/.local/share/dictate/tinyasr`).

## Build

`install.sh` does this for you. By hand, with Rust 1.85 or newer:

```sh
cd engine/worker
RUSTFLAGS="-C target-feature=+crt-static" cargo build --release --offline --locked
install -D -m 755 target/release/dictate-tinyasr ~/.local/lib/dictate/dictate-tinyasr
```

- There are no dependencies, so `--offline` works.
- `+crt-static` makes a static binary, which runs without any library from your system. On Fedora it
  needs `glibc-static`; on Debian and Ubuntu, `libc6-dev`.
- Do not add `-C target-cpu=native`: the engine picks the CPU code when it runs.
- Tests: `cargo test --offline` in `tinyasr/` and in `worker/`.

Without Rust, `install.sh` builds the same binary in a podman or docker image made from
[`Containerfile`](Containerfile) (Fedora's Rust; the build itself runs with no network).
