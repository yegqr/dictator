#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025-2026 The Dictator authors (see AUTHORS)
# Installs Dictator for this user only (no sudo):
#   ~/.local/bin/dictate, ~/.local/lib/dictate/parakeet.py      the helper program
#   ~/.local/lib/dictate/dictate-tinyasr                         the Small model's program (built here)
#   ~/.local/share/gnome-shell/extensions/<uuid>/                the GNOME extension
#   ~/.local/share/dictate/tinyasr/ or parakeet-tdt-0.6b-v3/     the speech model (asked: Small or Large)
# Options: --no-extension (you installed the extension from extensions.gnome.org), --yes (no questions:
#          the defaults, that is the Small model and your system's language)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UUID=dictator@kcharukovskyi.github.io
EXT=$HOME/.local/share/gnome-shell/extensions/$UUID
CONF=$HOME/.config/dictate
BIN=$HOME/.local/bin/dictate
LIB=$HOME/.local/lib/dictate
# the interpreter the helper runs with (its first line); GNOME starts it without your conda/venv
PY=/usr/bin/python3
# Small (tinyasr): 9 languages in 5 packs, every pack has English. A language comes from the first pack
# here that has it.
SMALL_PACKS=(en-ru-uk en-es-it-pt en-ru-uz en-fa en-km)
SMALL_LANGS="en ru uk es it pt uz fa km"
# Large (NVIDIA Parakeet): its 25 languages (helper/parakeet.py)
LARGE_LANGS="en de fr es it pt nl da sv fi pl cs sk sl hr hu ro lv lt et mt uk ru bg el"
WITH_EXT=1 YES=0
say() { printf '\n== %s\n' "$*"; }
ask() {   # ask "question" default(y|n)
  local a
  if [ "$YES" = 1 ]; then return 0; fi
  read -r -p "$1 $([ "$2" = y ] && echo '[Y/n]' || echo '[y/N]') " a
  a=${a:-$2}
  [[ $a == [yY]* ]]
}
answer() {   # answer "question" default: prints the reply (the default with --yes or an empty reply)
  local a=""
  if [ "$YES" != 1 ]; then read -r -p "$1 [$2] " a; fi
  echo "${a:-$2}"
}

# shellcheck source=/dev/null
. /etc/os-release 2>/dev/null || true
FAMILY=other
case " ${ID:-} ${ID_LIKE:-} " in
  *" fedora "*|*" rhel "*) FAMILY=fedora ;;
  *" ubuntu "*|*" debian "*) FAMILY=debian ;;
  *" arch "*) FAMILY=arch ;;
  *" opensuse"*|*" suse "*) FAMILY=suse ;;
esac
pkg() {   # pkg <fedora> <debian> <arch> <suse>: the install command for this distribution
  case $FAMILY in
    fedora) echo "sudo dnf install $1" ;;
    debian) echo "sudo apt install $2" ;;
    arch) echo "sudo pacman -S $3" ;;
    suse) echo "sudo zypper install $4" ;;
    *) echo "your package manager: $2 (Debian names)" ;;
  esac
}

# ~/.config/dictate/speech, the settings window's file (format: helper/dictate, extension/keys.js)
speech_get() { [ ! -f "$CONF/speech" ] || sed -n -E "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$CONF/speech" | tail -1; }
speech_set() {   # speech_set key value
  mkdir -p "$CONF"
  if grep -qE "^[[:space:]]*$1[[:space:]]*=" "$CONF/speech" 2>/dev/null; then
    sed -i -E "s/^[[:space:]]*$1[[:space:]]*=.*/$1 = $2/" "$CONF/speech"
  else
    echo "$1 = $2" >> "$CONF/speech"
  fi
}
only() {   # only "allowed codes" "codes": the allowed ones, each once, in order
  local c out=" " list
  read -r -a list <<< "${2//,/ }"
  for c in "${list[@]}"; do
    c=${c,,}
    if [[ " $1 " == *" $c "* && $out != *" $c "* ]]; then out+="$c "; fi
  done
  out=${out# }; echo "${out% }"
}
# packs_for "codes": the Small packs these languages need (other languages first, as every pack has
# English); en-ru-uk if none
packs_for() {
  local c p out=" " list
  read -r -a list <<< "$(only "${SMALL_LANGS/en /}" "$1") en"
  for c in "${list[@]}"; do
    for p in "${SMALL_PACKS[@]}"; do
      if [[ -$p- == *-$c-* ]]; then
        if [[ $out == " " || ($c != en && $out != *" $p "*) ]]; then out+="$p "; fi
        break
      fi
    done
  done
  out=${out# }; echo "${out% }"
}

check_system() {
  say "1/6 Checking the system"
  if command -v gnome-shell >/dev/null; then
    v=$(gnome-shell --version | grep -oE '[0-9]+' | head -1)
    tested=$(tr -d '\n' < "$HERE/extension/metadata.json" | grep -oE '"shell-version"[^]]*' | grep -oE '[0-9]+' | paste -sd ' ')
    echo "GNOME Shell $v"
    [[ " $tested " == *" $v "* ]] || echo "WARNING: the extension is made for GNOME $tested; GNOME $v will not load it."
  else
    echo "WARNING: GNOME Shell not found. Dictator is a GNOME extension."
  fi
  missing=()
  command -v pw-record >/dev/null && command -v pw-dump >/dev/null ||
    missing+=("pw-record, pw-dump: $(pkg pipewire-utils pipewire-bin pipewire pipewire-tools)")
  [ -x "$PY" ] || missing+=("$PY: $(pkg python3 python3 python python3)")
  if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing programs:"; printf '  %s\n' "${missing[@]}"
    echo "Install them, then run ./install.sh again."
    exit 1
  fi
  echo OK
}

# choose_model: sets MODEL (small or large) and LANGS_ON (the chosen languages first, then the others
# already on: each model uses the ones it has)
choose_model() {
  local cur def loc allowed langs c codes skipped=""
  say "2/6 Speech model and languages"
  cur=$(speech_get model)
  def=1; [[ $cur == int8 || $cur == fp32 ]] && def=2
  cat <<EOT
  1) Small (recommended): 9 languages, very fast, about 0.3 GB of memory, a 30 to 33 MB download per
     language pack. English with Russian and Ukrainian, Spanish, Italian, Portuguese, Uzbek, Persian or
     Khmer. It runs only in the bubblewrap sandbox, so it needs bubblewrap.
  2) Large: NVIDIA Parakeet, 25 European languages, slower, about 1.7 GB of memory, a 2.5 GB download.
EOT
  case $(answer "Model, 1 or 2?" $def) in 2*|[lL]*) MODEL=large ;; *) MODEL=small ;; esac
  allowed=$SMALL_LANGS; [ "$MODEL" = large ] && allowed=$LARGE_LANGS
  # suggested: the languages already set, else your system's language and English
  loc=${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}
  def=$(only "$allowed" "$(speech_get languages)")
  [ -n "$def" ] || def=$(only "$allowed" "${loc%%[_.@]*} en")
  echo "Languages of the $([ "$MODEL" = small ] && echo Small || echo Large) model: $allowed"
  read -r -a codes <<< "$(answer "Which do you speak? Codes, main first:" "$def" | tr ',' ' ')"
  for c in "${codes[@]}"; do [[ " $allowed " == *" ${c,,} "* ]] || skipped+=" $c"; done
  [ -z "$skipped" ] || echo "Not in this model, skipped:$skipped"
  langs=$(only "$allowed" "${codes[*]}"); [ -n "$langs" ] || langs=$def
  LANGS_ON=$langs
  read -r -a codes <<< "$(speech_get languages | tr ',' ' ')"
  for c in "${codes[@]}"; do [[ " $LANGS_ON " == *" $c "* ]] || LANGS_ON+=" $c"; done
  echo "Model: $MODEL. Languages: $LANGS_ON$([ "$MODEL" = small ] && echo ". Packs: $(packs_for "$langs")")"
}

save_choice() {   # once the model's program is ready: the model and the languages into the settings
  local cur
  cur=$(speech_get model)
  speech_set languages "$LANGS_ON"
  if [ "$MODEL" = small ]; then
    speech_set model small
  elif [[ $cur != int8 && $cur != fp32 ]]; then
    speech_set model int8
  fi
}

large_runtime() {   # where the Large model runs: the system Python, or a container image built here
  say "3/6 Where the speech model runs"
  mkdir -p "$CONF"
  if "$PY" -I -c 'import onnxruntime, numpy' 2>/dev/null; then
    echo host > "$CONF/runtime"
    echo "System Python has onnxruntime and numpy: the model runs with them."
    sandbox_note
    "$PY" -I -c 'import onnx' 2>/dev/null ||
      echo "Note: the Standard (8-bit) model is made on this computer and needs one more package: $(pkg python3-onnx python3-onnx python-onnx python3-onnx)"
  else
    ENGINE=""
    for e in podman docker; do command -v $e >/dev/null && { ENGINE=$e; break; }; done
    echo "System Python ($PY) has no onnxruntime + numpy. Either:"
    echo "  A) install them: $(pkg 'python3-onnxruntime python3-numpy python3-onnx' 'python3-onnxruntime python3-numpy python3-onnx' 'python-onnxruntime python-numpy python-onnx' 'python3-onnxruntime python3-numpy python3-onnx')"
    echo "     and run ./install.sh again, or"
    if [ -z "$ENGINE" ]; then
      echo "  B) install podman and run ./install.sh again."
      exit 1
    fi
    echo "  B) build a local $ENGINE image with them (Fedora packages, ~0.6 GB; the model runs in it without network)."
    ask "Build the image now (B)?" y || { echo "Stopped."; exit 1; }
    $ENGINE build -t dictate-asr -f "$HERE/helper/Containerfile" "$HERE/helper"
    echo "$ENGINE" > "$CONF/runtime"
  fi
}

sandbox_note() {
  if command -v bwrap >/dev/null; then
    echo "Sandbox: bubblewrap (no network, no access to your files)."
  else
    echo "Sandbox: unshare (no network only). For full isolation: $(pkg bubblewrap bubblewrap bubblewrap bubblewrap)"
  fi
}

small_runtime() {   # the Small model runs on this computer, only in bubblewrap; its program is built here
  say "3/6 Building the Small model's program"
  if ! command -v bwrap >/dev/null; then
    echo "The Small model runs only in the bubblewrap sandbox, and bubblewrap is not installed."
    echo "Install it: $(pkg bubblewrap bubblewrap bubblewrap bubblewrap)"
    echo "Then run ./install.sh again, or choose the Large model."
    exit 1
  fi
  echo "Sandbox: bubblewrap (no network, no access to your files)."
  if ! build_worker; then
    echo "Could not build it. Either install Rust ($(pkg 'cargo glibc-static' 'cargo libc6-dev' rust 'cargo glibc-devel-static'))"
    echo "or podman ($(pkg podman podman podman podman)) and run ./install.sh again, or choose the Large model."
    exit 1
  fi
}

# build_worker: builds engine/worker into $LIB/dictate-tinyasr, a static binary with no target-cpu (the
# engine picks the CPU code when it runs). The temp folder is removed also on Ctrl+C (the EXIT trap).
build_worker() {
  BUILD_TMP=$(mktemp -d) || return 1
  trap 'rm -rf "$BUILD_TMP"' EXIT
  if ! { worker_host "$BUILD_TMP" || worker_container "$BUILD_TMP"; }; then
    rm -rf "$BUILD_TMP"; trap - EXIT
    return 1
  fi
  install -D -m 755 "$BUILD_TMP/dictate-tinyasr" "$LIB/dictate-tinyasr" ||
    { echo "It was built, but could not be installed as $LIB/dictate-tinyasr."; exit 1; }
  rm -rf "$BUILD_TMP"; trap - EXIT
  echo "Small model's program: $LIB/dictate-tinyasr"
}

worker_host() {   # with this computer's Rust, offline (the engine has no dependencies)
  command -v cargo >/dev/null || return 1
  echo "Building with Rust, about a minute..."
  if (cd "$HERE/engine/worker" && CARGO_TARGET_DIR="$1/target" RUSTFLAGS="-C target-feature=+crt-static" \
        cargo build --release --offline --locked 2>&1 | tee "$1/cargo.log"); then
    mv "$1/target/release/dictate-tinyasr" "$1/" && rm -rf "$1/target"
  else
    echo "The build with Rust failed. Its last lines:"
    tail -n 12 "$1/cargo.log" | sed 's/^/  | /'
    if grep -qE 'cannot find -l|unable to find library -l' "$1/cargo.log"; then
      echo "The linker misses the C library's static files: $(pkg glibc-static libc6-dev glibc glibc-devel-static)"
    fi
    return 1
  fi
}

worker_container() {   # with Fedora's Rust in a podman or docker image (engine/Containerfile)
  local e ok=1
  for e in podman docker ""; do [ -n "$e" ] && command -v $e >/dev/null && break; done
  [ -n "$e" ] || return 1
  echo "Building in a $e container: Fedora's Rust (an image of about 0.8 GB, removed after; the Fedora base image stays), then the build with no network."
  # the source is read-only; the binary comes out on stdout, owned by you
  $e build -t dictate-tinyasr-build -f "$HERE/engine/Containerfile" "$HERE/engine" &&
  $e run --rm --network none --security-opt label=disable -v "$HERE/engine:/src:ro" -w /src/worker \
    -e CARGO_HOME=/tmp/cargo -e CARGO_TARGET_DIR=/tmp/target -e RUSTFLAGS="-C target-feature=+crt-static" \
    dictate-tinyasr-build sh -c 'cargo build --release --offline --locked >&2 && cat /tmp/target/release/dictate-tinyasr' \
    > "$1/dictate-tinyasr" && ok=0
  $e rmi dictate-tinyasr-build >/dev/null 2>&1 || true   # also after a failed build
  return $ok
}

install_files() {
  say "4/6 Installing"
  install -D -m 755 "$HERE/helper/dictate" "$BIN"
  install -D -m 644 "$HERE/helper/parakeet.py" "$LIB/parakeet.py"
  echo "Helper: $BIN"
  if [ "$WITH_EXT" = 1 ]; then
    mkdir -p "$EXT"
    install -m 644 "$HERE"/extension/{metadata.json,extension.js,core.js,keys.js,prefs.js,stylesheet.css,stylesheet-light.css} "$EXT/"
    echo "Extension: $EXT"
  fi
}

get_model() {
  say "5/6 Downloading the speech model"
  if [ "$MODEL" = small ]; then
    echo "Small: language packs $(packs_for "$(speech_get languages)"), a 30 to 33 MB download each, from tinyasr"
    echo "on GitHub (a fixed version, every file checked against its SHA-256), 35 to 38 MB each on disk."
    echo "(Or later: microphone menu -> Settings -> Speech, also for more languages.)"
    if ask "Download it now?" y; then
      "$BIN" --download small || echo "The model is not installed. Try again later: Settings -> Speech -> Speech Model."
    fi
  elif "$BIN" --models | "$PY" -I -c 'import json, sys; v = json.load(sys.stdin)["variants"]
sys.exit(0 if v["int8"]["installed"] or v["fp32"]["installed"] else 1)'; then
    echo "Installed."
  else
    echo "The Standard model: a 2.5 GB download from Hugging Face (NVIDIA Parakeet TDT 0.6B v3), converted"
    echo "to 8-bit on this computer (about a minute, 6 GB of free memory), then 0.9 GB on disk."
    echo "(Or later: microphone menu -> Settings -> Speech, also for the Full Precision model.)"
    if ask "Download it now?" y; then
      "$BIN" --download int8 || echo "The model is not installed. Try again later: Settings -> Speech -> Speech Model."
    fi
  fi
}

final_check() {
  say "6/6 Check"
  if "$BIN" --models | "$PY" -I -c 'import json, sys; sys.exit(0 if json.load(sys.stdin)["used"] else 1)'; then
    "$BIN" --selftest
  else
    echo "Skipped: no model yet."
  fi
  if [ "$WITH_EXT" = 1 ] && command -v gsettings >/dev/null; then
    cur=$(gsettings get org.gnome.shell enabled-extensions)
    if [[ $cur != *"'$UUID'"* ]]; then
      new=$("$PY" -I -c 'import ast, sys; l = ast.literal_eval(sys.argv[1].replace("@as ", "")); print(repr(l + [sys.argv[2]]))' "$cur" "$UUID")
      gsettings set org.gnome.shell enabled-extensions "$new"
    fi
  fi
  cat <<EOT

Done. Log out and log in once, so that GNOME loads the extension. Then:
  Ctrl+Alt+Space   start, speak     Enter   finish: the text is typed where your cursor was
  Caps Lock        pause / resume   Esc     cancel
Settings: the microphone icon in the top bar -> Settings.
EOT
}

main() {
  for a in "$@"; do
    case $a in
      --no-extension) WITH_EXT=0 ;;
      --yes|-y) YES=1 ;;
      *) echo "Unknown option: $a"; exit 2 ;;
    esac
  done
  check_system
  choose_model
  if [ "$MODEL" = small ]; then small_runtime; else large_runtime; fi
  save_choice
  install_files
  get_model
  final_check
}

# run, unless sourced (then the steps above can be run one by one)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
