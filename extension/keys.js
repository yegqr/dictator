// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 The Dictator authors (see AUTHORS)
// Settings files in ~/.config/dictate, shared with the helper program (helper/dictate), which reads
// them too: that is why they are plain files and not GSettings. Written by the settings window
// (prefs.js), read by the extension. Imports only GLib / Gio: prefs.js runs outside GNOME Shell.
//   keys          shortcuts, "name = GNOME accelerator" (see HEADER)
//   privacy       history = on|off, keep_minutes = N, clipboard = on|off, keep_audio = on|off, again = codes
//   speech        languages, text options, microphone, model... (see writeSpeech)
//   replacements  "what you say = what is typed", one per line
//   pill          where you dragged the recording indicator (see readPillSpot)
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';

const DIR = GLib.build_filenamev([GLib.get_user_config_dir(), 'dictate']);
export const KEYS_FILE = GLib.build_filenamev([DIR, 'keys']);
export const CAPS = 'CapsLock';
export const NONE = 'none';
export const SAME = 'start';            // finish = the start shortcut again
export const SHIFT_CAPS = 'Shift+CapsLock';  // finish = Caps Lock pressed while Shift is held
export const DEFAULTS = Object.freeze({start: '<Control><Alt>space', finish: 'Return', pause: CAPS, cancel: 'Escape'});

const HEADER = `# Dictator keys. Easiest to change in the settings window: top-bar microphone menu ->
# "Settings", or: gnome-extensions prefs dictator@kcharukovskyi.github.io
# By hand: name = GNOME accelerator, e.g. <Control><Alt>space  <Super>d  <Control><Shift>d  F9  Pause
#   start   start a recording (repeat the line for more shortcuts)
#   finish  while recording: finish and paste (Return = Enter); start = the start shortcut again;
#           Shift+CapsLock = Caps Lock pressed while Shift is held (the lock is not toggled)
#   pause   pause / resume while recording; CapsLock = Caps Lock (the lock is not toggled); none = off
#   cancel  cancel while recording; none = off
# Applies at once, no log out. A shortcut that GNOME or another app already uses does not work.
`;

// Where the read*() functions get the files' text from: in GNOME Shell, from SettingsCache (read
// asynchronously, kept in memory); in the settings window, which runs in its own process, prefs.js
// sets a reader that reads the file. The reader returns the text, or null when there is no file.
let reader = () => null;

export function setReader(fn) {
    reader = fn ?? (() => null);
}

function load(path) {
    return reader(path);
}

const WATCHED = () => [KEYS_FILE, PRIVACY_FILE, SPEECH_FILE, PILL_FILE];

// The settings files for GNOME Shell: read asynchronously at start and again whenever one changes
// (the settings window writes them), kept in memory for the read*() functions. onChange(path) is
// called once a file's new text is in.
export class SettingsCache {
    constructor(onChange) {
        this._onChange = onChange;
        this._texts = new Map();
        this._seq = new Map();
    }

    start() {
        this._cancellable = new Gio.Cancellable();
        setReader(path => this._texts.get(path) ?? null);
        const dir = Gio.File.new_for_path(DIR);
        try {
            dir.make_directory_with_parents(null);
        } catch (e) {}                  // already there
        try {
            this._monitor = dir.monitor_directory(Gio.FileMonitorFlags.WATCH_MOVES, null);
            this._monitorId = this._monitor.connect('changed', (m, file, other) => {
                for (const f of [file, other]) {
                    if (f && WATCHED().includes(f.get_path()))
                        this._load(f.get_path());
                }
            });
        } catch (e) {
            console.warn(`dictator: cannot watch ${DIR}: ${e}`);
        }
        for (const path of WATCHED())
            this._load(path);
    }

    // whether the file was read at least once
    has(path) {
        return this._texts.has(path);
    }

    _load(path) {
        const seq = (this._seq.get(path) ?? 0) + 1;     // a save is several events: only the last counts
        this._seq.set(path, seq);
        Gio.File.new_for_path(path).load_contents_async(this._cancellable, (f, res) => {
            let text = null;
            try {
                const [, bytes] = f.load_contents_finish(res);
                text = new TextDecoder().decode(bytes);
            } catch (e) {
                if (e.matches(Gio.IOErrorEnum, Gio.IOErrorEnum.CANCELLED))
                    return;
            }
            if (this._seq.get(path) !== seq || (this._texts.has(path) && this._texts.get(path) === text))
                return;
            this._texts.set(path, text);
            this._onChange?.(path);
        });
    }

    stop() {
        this._cancellable?.cancel();
        this._cancellable = null;
        if (this._monitorId)
            this._monitor.disconnect(this._monitorId);
        this._monitorId = 0;
        this._monitor?.cancel();
        this._monitor = null;
        this._texts.clear();
        this._seq.clear();
        setReader(null);
    }
}

export function readKeys() {
    const keys = {start: [], finish: DEFAULTS.finish, pause: DEFAULTS.pause, cancel: DEFAULTS.cancel};
    const text = load(KEYS_FILE);
    if (text !== null) {
        for (const line of text.split('\n')) {
            const m = line.trim().match(/^(start|finish|pause|cancel)\s*=\s*(\S+)$/);
            if (!m)
                continue;
            if (m[1] !== 'start')
                keys[m[1]] = m[2];
            else if (m[2] !== NONE && !keys.start.includes(m[2]))
                keys.start.push(m[2]);
        }
    }
    if (!keys.start.length)
        keys.start = [DEFAULTS.start];
    if (keys.finish === NONE)               // there must be a way to finish
        keys.finish = DEFAULTS.finish;
    return keys;
}

export function writeKeys(keys) {
    GLib.mkdir_with_parents(DIR, 0o700);
    const lines = [...keys.start.map(a => `start = ${a}`), `finish = ${keys.finish}`, `pause = ${keys.pause}`,
        `cancel = ${keys.cancel}`];
    GLib.file_set_contents(KEYS_FILE, `${HEADER}${lines.join('\n')}\n`);
}

// GNOME's own shortcuts (and IBus's) always win over an extension's, silently: the extension "grabs"
// the combination without an error, but never gets the key. So a clash is checked here, by name.
const SCHEMAS = ['org.gnome.desktop.wm.keybindings', 'org.gnome.shell.keybindings', 'org.gnome.mutter.keybindings',
    'org.gnome.mutter.wayland.keybindings', 'org.gnome.settings-daemon.plugins.media-keys',
    'org.freedesktop.ibus.general.hotkey'];
const CUSTOM = 'org.gnome.settings-daemon.plugins.media-keys.custom-keybinding';

// "<Primary><Alt>Space" and "<Alt><Control>space" are the same shortcut
export function normAccel(a) {
    const alias = {primary: 'control', ctrl: 'control', ctl: 'control', mod1: 'alt', mod4: 'super'};
    const mods = new Set();
    const key = a.replace(/<([^>]+)>/g, (m, n) => {
        mods.add(alias[n.toLowerCase()] ?? n.toLowerCase());
        return '';
    });
    return `${[...mods].sort().join('+')}|${key.toLowerCase()}`;
}

// [[accelerator, what uses it], ...]
export function gnomeShortcuts() {
    const out = [];
    const src = Gio.SettingsSchemaSource.get_default();
    const nice = k => k.replace(/-/g, ' ').replace(/^./, c => c.toUpperCase());
    for (const id of SCHEMAS) {
        const schema = src?.lookup(id, true);
        if (!schema)
            continue;
        const s = Gio.Settings.new_full(schema, null, null);
        for (const k of schema.list_keys()) {
            if (k === 'custom-keybindings' || schema.get_key(k).get_value_type().dup_string() !== 'as')
                continue;
            for (const a of s.get_strv(k))
                out.push([a, `${id.includes('ibus') ? 'IBus' : 'GNOME'} (“${nice(k)}”)`]);
        }
        const custom = src.lookup(CUSTOM, true);
        if (id.endsWith('media-keys') && custom) {
            for (const path of s.get_strv('custom-keybindings')) {
                const c = Gio.Settings.new_full(custom, null, path);
                out.push([c.get_string('binding'), `your custom shortcut “${c.get_string('name')}”`]);
            }
        }
    }
    return out;
}

// what already uses this shortcut, or null
export function takenBy(accel, list = gnomeShortcuts()) {
    const n = normAccel(accel);
    return list.find(([a]) => a && normAccel(a) === n)?.[1] ?? null;
}

export const PRIVACY_FILE = GLib.build_filenamev([DIR, 'privacy']);
// again: the languages offered by "Transcribe Again" in the menu; null = the enabled languages
export const PRIVACY_DEFAULTS = Object.freeze({history: true, keepMinutes: 60, clipboard: false, keepAudio: false,
    again: null});

export function readPrivacy() {
    const p = {...PRIVACY_DEFAULTS};
    const codes = new Set(LANGUAGES.map(([c]) => c));
    for (const line of (load(PRIVACY_FILE) ?? '').split('\n')) {
        const m = line.trim().match(/^(history|keep_minutes|clipboard|keep_audio|again)\s*=\s*(.*)$/);
        const v = m?.[2].trim();
        if (m?.[1] === 'history')
            p.history = v !== 'off';
        else if (m?.[1] === 'clipboard')
            p.clipboard = v === 'on';
        else if (m?.[1] === 'keep_audio')
            p.keepAudio = v === 'on';
        else if (m?.[1] === 'again')
            p.again = [...new Set(v.split(/[\s,]+/).filter(c => codes.has(c)))];
        else if (m && /^\d+$/.test(v))
            p.keepMinutes = parseInt(v);
    }
    return p;
}

// the languages "Transcribe Again" offers: chosen in Settings, else the enabled ones
export function againLanguages(p = readPrivacy()) {
    return p.again ?? readSpeech().languages;
}

export function writePrivacy(p) {
    GLib.mkdir_with_parents(DIR, 0o700);
    GLib.file_set_contents(PRIVACY_FILE,
        '# Dictator privacy (settings window: microphone menu -> "Settings").\n' +
        '# history = on|off: keep the last 3 texts (in RAM only) for the microphone menu\n' +
        '# keep_minutes = N: delete kept recordings (also failed ones) after N minutes; 0 = until restart\n' +
        '# clipboard = on|off: apps that cannot take typed text get it by a paste through the clipboard;\n' +
        '#   off: the clipboard is never used, the text waits in the microphone menu\n' +
        '# keep_audio = on|off: the history keeps the audio too (RAM only), for "Transcribe Again"\n' +
        '# again = codes: the languages "Transcribe Again" offers (no line: the enabled languages)\n' +
        `history = ${p.history ? 'on' : 'off'}\nkeep_minutes = ${p.keepMinutes}\n` +
        `clipboard = ${p.clipboard ? 'on' : 'off'}\nkeep_audio = ${p.keepAudio ? 'on' : 'off'}\n` +
        (p.again ? `again = ${p.again.join(' ')}\n` : ''));
}

// Speech settings; the helper reads all but space_after and pill.
export const SPEECH_FILE = GLib.build_filenamev([DIR, 'speech']);
export const REPLACE_FILE = GLib.build_filenamev([DIR, 'replacements']);
// every language a speech model knows: [code, English name, own name]. The large model (Parakeet) has
// all but SMALL_ONLY; the small one (tinyasr) has those of its language packs (helper: dictate --models).
export const LANGUAGES = Object.freeze([
    ['uk', 'Ukrainian', 'Українська'], ['en', 'English', 'English'], ['bg', 'Bulgarian', 'Български'],
    ['hr', 'Croatian', 'Hrvatski'], ['cs', 'Czech', 'Čeština'], ['da', 'Danish', 'Dansk'],
    ['nl', 'Dutch', 'Nederlands'], ['et', 'Estonian', 'Eesti'], ['fi', 'Finnish', 'Suomi'],
    ['fr', 'French', 'Français'], ['de', 'German', 'Deutsch'], ['el', 'Greek', 'Ελληνικά'],
    ['hu', 'Hungarian', 'Magyar'], ['it', 'Italian', 'Italiano'], ['km', 'Khmer', 'ខ្មែរ'],
    ['lv', 'Latvian', 'Latviešu'], ['lt', 'Lithuanian', 'Lietuvių'], ['mt', 'Maltese', 'Malti'],
    ['fa', 'Persian', 'فارسی'], ['pl', 'Polish', 'Polski'], ['pt', 'Portuguese', 'Português'],
    ['ro', 'Romanian', 'Română'], ['ru', 'Russian', 'Русский'], ['sk', 'Slovak', 'Slovenčina'],
    ['sl', 'Slovenian', 'Slovenščina'], ['es', 'Spanish', 'Español'], ['sv', 'Swedish', 'Svenska'],
    ['uz', 'Uzbek', 'Oʻzbekcha'],
]);
export const SMALL_ONLY = Object.freeze(['uz', 'fa', 'km']);
// the speech model setting: small (tinyasr), or large (Parakeet) as int8 or fp32
export const MODELS = Object.freeze(['small', 'int8', 'fp32']);
export const SPEECH_DEFAULTS = Object.freeze({languages: ['uk', 'en', 'ru'], margin: 0.15, threads: 0,
    finalPeriod: 'keep', spaceAfter: false, microphone: 'default', autoStop: 0, voiceGate: 'off', pill: 'bottom',
    spokenPunctuation: false, fillers: true, model: 'small'});
// a line break the user dictated ("новий рядок"): the helper sends this private-use character, the
// extension types a line break, or a space in a terminal (there a line break would run a command)
export const NEWLINE_MARK = '\ue000';

export function readSpeech() {
    const p = {...SPEECH_DEFAULTS, languages: [...SPEECH_DEFAULTS.languages]};
    const codes = new Set(LANGUAGES.map(([c]) => c));
    for (const line of (load(SPEECH_FILE) ?? '').split('\n')) {
        const m = line.trim().match(
            /^(languages|margin|threads|final_period|space_after|microphone|auto_stop|voice_gate|pill|spoken_punctuation|fillers|model)\s*=\s*(.*)$/);
        if (!m)
            continue;
        const v = m[2].trim();
        if (m[1] === 'languages') {
            const l = [...new Set(v.split(/[\s,]+/).filter(c => codes.has(c)))];
            if (l.length)
                p.languages = l;
        } else if (m[1] === 'margin' && Number.isFinite(parseFloat(v))) {
            p.margin = Math.min(2, Math.max(0, parseFloat(v)));
        } else if (m[1] === 'threads' && /^\d+$/.test(v)) {
            p.threads = Math.min(64, parseInt(v));
        } else if (m[1] === 'final_period' && ['keep', 'drop', 'chats'].includes(v)) {
            p.finalPeriod = v;
        } else if (m[1] === 'space_after') {
            p.spaceAfter = v === 'on';
        } else if (m[1] === 'microphone' && /^[\w.:@+-]+$/.test(v)) {
            p.microphone = v;
        } else if (m[1] === 'auto_stop' && /^\d+$/.test(v)) {
            p.autoStop = Math.min(60, parseInt(v));
        } else if (m[1] === 'voice_gate' && ['off', 'light', 'medium', 'strong'].includes(v)) {
            p.voiceGate = v;
        } else if (m[1] === 'pill' && ['bottom', 'top'].includes(v)) {
            p.pill = v;
        } else if (m[1] === 'spoken_punctuation') {
            p.spokenPunctuation = v === 'on';
        } else if (m[1] === 'fillers') {
            p.fillers = v !== 'off';
        } else if (m[1] === 'model' && MODELS.includes(v)) {
            p.model = v;
        }
    }
    return p;
}

export function writeSpeech(p) {
    GLib.mkdir_with_parents(DIR, 0o700);
    GLib.file_set_contents(SPEECH_FILE,
        '# Dictator speech settings (settings window: microphone menu -> "Settings", page "Speech").\n' +
        '# languages: enabled, main first (codes: ' + LANGUAGES.map(([c]) => c).join(' ') + ')\n' +
        '# margin: how much surer another Cyrillic language must be than the main one (0.15)\n' +
        '# threads: CPU threads for the model, 0 = automatic; final_period: keep|drop|chats; space_after: on|off\n' +
        '# spoken_punctuation: on|off ("кома", "знак питання", "новий рядок"...); fillers: on|off (remove "еее", "um")\n' +
        '# microphone: default or a PipeWire source name (node.name in pw-dump); auto_stop: seconds of silence, 0 = off\n' +
        '# voice_gate: off|light|medium|strong (mute sound much quieter than your voice); pill: bottom|top\n' +
        '# model: small|int8|fp32 (small: tinyasr; int8, fp32: Parakeet; if not installed, another one that is)\n' +
        `languages = ${p.languages.join(' ')}\nmargin = ${p.margin}\nthreads = ${p.threads}\n` +
        `final_period = ${p.finalPeriod}\nspace_after = ${p.spaceAfter ? 'on' : 'off'}\n` +
        `microphone = ${p.microphone}\nauto_stop = ${p.autoStop}\nvoice_gate = ${p.voiceGate}\npill = ${p.pill}\n` +
        `spoken_punctuation = ${p.spokenPunctuation ? 'on' : 'off'}\nfillers = ${p.fillers ? 'on' : 'off'}\n` +
        `model = ${p.model}\n`);
}

// Where you dragged the recording indicator: its centre as a share of the monitor's width and height
// (0..1), so it lands in the same spot on any monitor. null: the place chosen in Settings.
export const PILL_FILE = GLib.build_filenamev([DIR, 'pill']);

export function readPillSpot() {
    const m = (load(PILL_FILE) ?? '').match(/^\s*x\s*=\s*([\d.]+)\s*\n\s*y\s*=\s*([\d.]+)\s*$/m);
    const [x, y] = m ? [parseFloat(m[1]), parseFloat(m[2])] : [NaN, NaN];
    return x >= 0 && x <= 1 && y >= 0 && y <= 1 ? {x, y} : null;
}

// spot: {x, y} as above, or null to go back to the place chosen in Settings
// asynchronous (it is written from GNOME Shell, when you drop the pill); null forgets the place
export function writePillSpot(spot) {
    const file = Gio.File.new_for_path(PILL_FILE);
    if (!spot) {
        file.delete_async(GLib.PRIORITY_DEFAULT, null, (f, res) => {
            try {
                f.delete_finish(res);
            } catch (e) {}              // there was none
        });
        return;
    }
    const text = '# Dictator: where you dragged the recording indicator (share of the screen)\n' +
        `x = ${spot.x.toFixed(4)}\ny = ${spot.y.toFixed(4)}\n`;
    file.replace_contents_bytes_async(new GLib.Bytes(new TextEncoder().encode(text)), null, false,
        Gio.FileCreateFlags.PRIVATE | Gio.FileCreateFlags.REPLACE_DESTINATION, null, (f, res) => {
            try {
                f.replace_contents_finish(res);
            } catch (e) {
                console.warn(`dictator: cannot save the pill position: ${e.message}`);
            }
        });
}

// the replacements file, created with an explanation the first time
export function ensureReplacements() {
    if (GLib.file_test(REPLACE_FILE, GLib.FileTest.EXISTS))
        return REPLACE_FILE;
    GLib.mkdir_with_parents(DIR, 0o700);
    GLib.file_set_contents(REPLACE_FILE,
        '# Dictator word replacements and snippets: one per line, "what you say = what is typed".\n' +
        '# Whole words only (so word forms need their own lines), upper/lower case does not matter.\n' +
        '# Examples:\n# пайтон = Python\n# гіт хаб = GitHub\n# мій імейл = me@example.com\n' +
        '# моє посилання = https://example.com/me\n');
    return REPLACE_FILE;
}

// Speech gives one line of words: control characters (line breaks, escape sequences), zero-width and
// bidi-override characters are cut (the helper does it too), so a text can never run a command in a
// terminal or hide text.
const BAD_CHARS = /[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069\ufeff]/g;
const MAX_TEXT = 50000;

export function cleanText(t) {
    return typeof t === 'string' ? t.replace(BAD_CHARS, ' ').replace(/\s+/g, ' ').trim().slice(0, MAX_TEXT) : '';
}

// "<Control><Alt>space" -> "Ctrl+Alt+Space" (for messages on screen)
export function prettyAccel(a) {
    if (a === CAPS)
        return 'Caps Lock';
    if (a === SHIFT_CAPS)
        return 'Shift+Caps Lock';
    if (a === NONE)
        return 'off';
    const mods = {control: 'Ctrl', primary: 'Ctrl', ctrl: 'Ctrl', alt: 'Alt', super: 'Super', shift: 'Shift',
        meta: 'Meta', hyper: 'Hyper'};
    const names = {Return: 'Enter', KP_Enter: 'Enter', Escape: 'Esc', Caps_Lock: 'Caps Lock', BackSpace: 'Backspace'};
    const key = a.replace(/<([^>]+)>/g, (m, n) => `${mods[n.toLowerCase()] ?? n}+`);
    return key.replace(/[^+]+$/, k => names[k] ?? (k.length === 1 ? k.toUpperCase() : k[0].toUpperCase() + k.slice(1)));
}
