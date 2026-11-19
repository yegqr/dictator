# Notes for extensions.gnome.org reviewers

**What it does.** Offline voice dictation. The extension draws the UI: a pill at the bottom of the
screen, a top-bar icon with a menu, and the preferences window. It talks to a separate helper program
over stdin and stdout. The helper records with `pw-record` and runs a speech model in a sandbox
without network access.

**Why a helper outside the extension.** Speech recognition needs native code and a model of 35 MB to
1 GB on disk: the Small model is a Rust program (`engine/`, built from source on the user's computer
by `install.sh`). The Large model needs ONNX Runtime and NumPy, which have no GJS bindings.
- The helper is readable Python, licensed GPL-3.0-or-later, and runs unprivileged:
  https://github.com/kcharukovskyi/dictator/tree/main/helper
- The extension does not install or download the helper. The user installs it from the repository
  with `install.sh`. Without it, the extension says so and links to the instructions.

**Subprocesses.**
- `core.js` spawns `~/.local/bin/dictate` with `Gio.Subprocess`, only when the user starts a recording
  or presses Retry in the menu. The process exits when the recording is finished or cancelled.
- `disable()` cancels a running recording and calls `force_exit()` on a running Retry.
- `kitten @ ls/focus-window` (kitty's own CLI) is run only if kitty is installed, to return to the
  right tab. It is killed after 3 s.
- `prefs.js` runs `dictate --models`, `--download` and `--remove` for the Speech Model section, and
  `dictate --try` for Try Dictation. That is a test recording whose text is shown in the window and
  is neither typed nor kept. Closing the window cancels it.

**Network.** The extension has no network access.
- The model download is in the helper. It runs only after the user clicks Download in preferences
  (or answers yes in `install.sh`).
- It fetches fixed files, from a pinned Hugging Face revision (Large) or a pinned GitHub commit of
  tinyasr (Small), and checks each one against a hard-coded SHA-256.
- These are model weights, not code.
- Closing the preferences window stops the download.

**Why `unlock-dialog`.** If the screen locks during a recording, disabling the extension would kill the
helper and lose what the user said. With `unlock-dialog`, locking finishes and transcribes the
recording, and the text waits in memory until the unlock. While the screen is locked:
- Every keyboard signal and accelerator grab is disconnected: `_disconnectKeys()` in `extension.js`
  and `_disconnectKeyboard()` in `core.js`.
- The indicator and the pill are hidden, and nothing is typed.

Everything is reconnected on unlock. `disable()` releases everything in either session mode (see the
comment in `extension.js`).

**Typing text.**
- Text goes in with `Main.inputMethod.commit()`, like the on-screen keyboard. It goes only into the
  window that had the focus when the recording started.
- The text is sanitised first: control, bidi and zero-width characters are removed.
- A virtual keyboard (`Clutter.Seat.create_virtual_device`) is used only to put Caps Lock back after it
  served as the pause key, and for Ctrl+V when the user has turned on clipboard pasting in
  preferences.

**Clipboard.** Declared in the description. The clipboard is written only in two cases:
- when the user clicks a past text in the top-bar menu (copy);
- when the user has turned on "Paste Through the Clipboard" (off by default), for apps without text
  input support.

In the second case the previous clipboard content is restored afterwards, and password-manager entries
are never put back. No default shortcut acts on the clipboard.

**The pill takes mouse input.** It is reactive so that it can be dragged. While the button is held,
`global.stage.grab()` sends the motion to it; the grab is dismissed on release, on lock and in
`disable()`. Its position is saved in `~/.config/dictate/pill` (two numbers).

**Shift+Caps Lock.** Not grabbed: the Caps Lock watch that already exists for the pause key checks
whether Shift is held (`global.get_pointer()` modifiers) when the lock changes, only while recording.

**Kept audio.** Off by default. With "Keep Audio to Transcribe Again" on, the helper moves the last 3
recordings into `$XDG_RUNTIME_DIR/dictate-history` (RAM) instead of deleting them; they expire with the
other kept recordings, and turning the switch off deletes them. `dictate --redo` gets only a stamp
checked against `rec-\d{8}-\d{6}` and a language code from a fixed list, as separate arguments.

**Monkey-patching.** The `showOSD` method of the quick-settings output slider instance is replaced
while recording. A Bluetooth headset switches profile when the microphone opens, and GNOME would flash
the volume OSD several times. `disable()` restores the method (`_unpatchOsd`).

**Settings storage.** Settings are plain files in `~/.config/dictate` rather than GSettings, because
the Python helper reads the same files without GObject bindings. `prefs.js` writes them.
- **No synchronous file IO in the Shell.** In the Shell, `SettingsCache` (`keys.js`) reads the files
  with `load_contents_async()` at enable and again whenever the directory monitor reports a change.
  It keeps the text in memory for the `read*()` functions. Shortcut changes therefore apply at once.
- **Other files.** The recent texts in the menu are read asynchronously when the menu opens. The pill
  position is written with `replace_contents_bytes_async()`.
- **`prefs.js`.** It runs in its own process and reads the files directly.

**Gio.Settings.** The extension has no settings schema of its own. `keys.js` opens GNOME's own
keybinding schemas read-only (`org.gnome.desktop.wm.keybindings`, media keys, IBus…). It does this
only to warn when a chosen shortcut is already taken, because GNOME silently keeps such keys.

**shell-version.** Only `51`, the stable release this was tested on.

**Logging.** Only warnings and errors are logged. Per-dictation notes use `console.debug` and never
contain the text.
