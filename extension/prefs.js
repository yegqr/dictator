// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025-2026 The Dictator authors (see AUTHORS)
// Settings window (top-bar microphone menu -> Settings). Two pages:
//   General  shortcuts, privacy, appearance
//   Speech   the speech model (Small with its language packs, or Large; download / remove through the
//            helper), languages, text, recording
// Settings are saved in ~/.config/dictate (see keys.js).
import Adw from 'gi://Adw';
import Gdk from 'gi://Gdk';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Gtk from 'gi://Gtk';
import {ExtensionPreferences} from 'resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js';
import {setReader, readKeys, writeKeys, readPrivacy, writePrivacy, readSpeech, writeSpeech, ensureReplacements, readPillSpot,
    writePillSpot, againLanguages, prettyAccel, gnomeShortcuts, NEWLINE_MARK, LANGUAGES, CAPS, NONE, SAME, SHIFT_CAPS,
    DEFAULTS, SMALL_ONLY} from './keys.js';

const THREAD_CHOICES = [[0, 'Automatic'], [1, '1'], [2, '2'], [4, '4'], [6, '6'], [8, '8']];
const STOP_CHOICES = [[0, 'Off'], [2, 'After 2 seconds'], [3, 'After 3 seconds'], [5, 'After 5 seconds'],
    [10, 'After 10 seconds']];
const GATE_CHOICES = [['off', 'Off'], ['light', 'Light'], ['medium', 'Medium'], ['strong', 'Strong']];
const PILL_CHOICES = [['bottom', 'Bottom'], ['top', 'Top']];
const HELPER = GLib.build_filenamev([GLib.get_home_dir(), '.local', 'bin', 'dictate']);
const REPO = 'https://github.com/kcharukovskyi/dictator';
const VARIANTS = [
    {id: 'int8', title: 'Standard', note: v => `The better of the two. Takes ${gb(v.size)}. Made on this computer from a ` +
        `${gb(v.download)} download, which needs ${v.memory_gb} GB of free memory for about a minute`},
    {id: 'fp32', title: 'Full Precision', note: v => `Slightly more accurate, about half as fast. ` +
        `${gb(v.download)} download, takes ${gb(v.size)}`},
];

const gb = bytes => `${(bytes / 1e9).toFixed(1)} GB`;
const mb = bytes => `${Math.round(bytes / 1e6)} MB`;

// English names; the large model's languages (the small one's come from its packs, see smallCodes)
const NAMES = new Map(LANGUAGES.map(([c, en]) => [c, en]));
const LARGE_CODES = new Set(LANGUAGES.map(([c]) => c).filter(c => !SMALL_ONLY.includes(c)));
// a pack's languages in words: "English, Russian, Ukrainian"
const inWords = codes => codes.map(c => NAMES.get(c) ?? c).join(', ');
// the model the settings show: the one in use, else the chosen one
const current = info => info?.used ?? info?.chosen ?? 'small';
// the small model's languages: those of its packs
const smallCodes = info => new Set(Object.values(info.small.packs ?? {}).flatMap(p => p.languages ?? []));

// what the helper says about the model files (see helper/dictate --models), or null without the helper
function modelInfo() {
    if (!GLib.file_test(HELPER, GLib.FileTest.IS_EXECUTABLE))
        return null;
    try {
        const proc = Gio.Subprocess.new([HELPER, '--models'],
            Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_SILENCE);
        const [, out] = proc.communicate_utf8(null, null);
        return JSON.parse(out);
    } catch (e) {
        return null;
    }
}

// [[source name, description]] of the microphones, from PipeWire
function microphones() {
    try {
        const proc = Gio.Subprocess.new(['pw-dump'],
            Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_SILENCE);
        const [, out] = proc.communicate_utf8(null, null);
        return JSON.parse(out).map(o => o.info?.props ?? {})
            .filter(p => p['media.class'] === 'Audio/Source' && p['node.name'])
            .map(p => [p['node.name'], p['node.description'] || p['node.name']]);
    } catch (e) {
        return [];
    }
}

// a ComboRow over [[value, label], ...]; an unknown current value is shown as-is at the end
function choiceRow(props, choices, current, onChange) {
    const row = new Adw.ComboRow({...props, model: Gtk.StringList.new(choices.map(([, l]) => l))});
    const list = [...choices];
    let i = list.findIndex(([v]) => v === current);
    if (i < 0) {
        list.push([current, String(current)]);
        row.model.append(String(current));
        i = list.length - 1;
    }
    row.selected = i;
    row.connect('notify::selected', () => onChange(list[row.selected]?.[0] ?? current));
    return row;
}

// how long kept recordings stay (minutes; 0 = until a restart)
const KEEP_CHOICES = [[15, '15 minutes'], [60, '1 hour'], [480, '8 hours'], [0, 'Until a restart']];

const ROWS = [
    {id: 'start', title: 'Start', hint: 'Starts recording'},
    {id: 'finish', title: 'Finish and Paste', hint: 'While recording (Enter also means the keypad Enter)', canSame: true,
        shiftCaps: true},
    {id: 'pause', title: 'Pause / Resume', hint: 'While recording', canOff: true, caps: true},
    {id: 'cancel', title: 'Cancel', hint: 'While recording', canOff: true},
];
// keys that may be a global shortcut on their own (they type nothing); others need Ctrl, Alt or Super
const BARE_OK = /^(F\d{1,2}|Pause|Scroll_Lock|Insert|Menu|Print|XF86\w+)$/;
const MODIFIER_KEY = /^((Shift|Control|Alt|Meta|Super|Hyper)_[LR]|ISO_Level\d_(Shift|Latch|Lock)|Num_Lock|ISO_Next_Group)$/;
const SPECIAL = [CAPS, NONE, SAME, SHIFT_CAPS];     // key values that are not GNOME accelerators
const CLOSE_MS = 400;           // the capture window stays this long after a key, taking what comes with it
function parse(accel) {
    const [ok, key, mods] = Gtk.accelerator_parse(accel);
    return ok && key ? [key, mods] : null;
}

function same(a, b) {
    if (a === b)
        return true;
    const pa = parse(a), pb = parse(b);
    return !!pa && !!pb && pa[0] === pb[0] && pa[1] === pb[1];
}

// row subtitles, toasts and status descriptions are Pango markup: plain text (an accelerator like
// <Super>d, an error message) must be escaped
const esc = text => GLib.markup_escape_text(text, -1);
const toast = (window, title) => window.add_toast(new Adw.Toast({title: esc(title), timeout: 5}));

// the settings window runs in its own process, not in GNOME Shell: it reads the settings files directly
function readFile(path) {
    try {
        return new TextDecoder().decode(GLib.file_get_contents(path)[1]);
    } catch (e) {
        return null;
    }
}

export default class DictatorPrefs extends ExtensionPreferences {
    fillPreferencesWindow(window) {
        setReader(readFile);
        window.set_default_size(640, 640);
        window.set_search_enabled(true);
        const page = new Adw.PreferencesPage({title: 'General', icon_name: 'preferences-system-symbolic'});
        const group = new Adw.PreferencesGroup({
            title: 'Keyboard Shortcuts',
            description: 'Applied at once, no need to log out. Saved in ~/.config/dictate/keys (you can edit it by hand too).',
        });
        page.add(group);
        window.add(page);
        const taken = gnomeShortcuts();
        const ui = {};

        const refresh = () => {
            const keys = readKeys();
            for (const r of ROWS) {
                const v = r.id === 'start' ? keys.start[0] : keys[r.id];
                const u = ui[r.id];
                const real = !SPECIAL.includes(v) && !!parse(v);
                u.accel.visible = real;
                u.accel.accelerator = real ? v : '';
                u.text.visible = !real;
                u.text.label = v === NONE ? 'Off' : v === SAME ? `${prettyAccel(keys.start[0])} again`
                    : SPECIAL.includes(v) ? prettyAccel(v) : v;
                const notes = [r.hint];
                if (r.id === 'start' && keys.start.length > 1)
                    notes.push(`Also (from the file): ${keys.start.slice(1).map(prettyAccel).join(', ')}`);
                const clash = real && taken.find(([a]) => same(a, v));
                if (clash)
                    notes.push(`⚠ Already used by ${clash[1]}: will not work, choose another`);
                u.row.subtitle = esc(notes.join('\n'));
                u.reset.sensitive = v !== DEFAULTS[r.id];
            }
        };

        // save one key; returns an error text or null
        const set = (id, value) => {
            const keys = readKeys();
            if (value !== NONE && value !== SAME) {
                for (const o of ROWS) {
                    if (o.id === id)
                        continue;
                    const used = o.id === 'start' ? keys.start : [keys[o.id]];
                    if (used.some(x => x !== NONE && x !== SAME && same(x, value)))
                        return `Already used for “${o.title}”. One key, one job.`;
                }
            }
            if (!SPECIAL.includes(value)) {
                const clash = taken.find(([a]) => same(a, value));
                if (clash)
                    return `${prettyAccel(value)} is already used by ${clash[1]}: it would never reach dictation. Choose another one.`;
            }
            if (id === 'start')
                keys.start[0] = value;
            else
                keys[id] = value;
            try {
                writeKeys(keys);
            } catch (e) {
                return `Cannot save: ${e.message}`;
            }
            refresh();
            return null;
        };

        for (const r of ROWS) {
            const row = new Adw.ActionRow({title: r.title, subtitle: r.hint});
            const accel = new Gtk.ShortcutLabel({valign: Gtk.Align.CENTER});
            const text = new Gtk.Label({valign: Gtk.Align.CENTER, css_classes: ['dim-label']});
            const change = new Gtk.Button({label: 'Change…', valign: Gtk.Align.CENTER});
            const reset = new Gtk.Button({icon_name: 'edit-undo-symbolic', valign: Gtk.Align.CENTER,
                tooltip_text: 'Back to the default', css_classes: ['flat']});
            change.connect('clicked', () => this._capture(window, r, set));
            reset.connect('clicked', () => set(r.id, DEFAULTS[r.id]));
            for (const w of [accel, text, change, reset])
                row.add_suffix(w);
            row.activatable_widget = change;
            group.add(row);
            ui[r.id] = {row, accel, text, reset};
        }
        refresh();
        page.add(this._privacyGroup());
        page.add(this._appearanceGroup());
        page.add(this._aboutGroup());
        const [speech, ready] = this._speechPage(window);
        window.add(speech);
        if (!ready)                 // the model (or the helper) is still missing: start there
            window.set_visible_page(speech);
    }

    _appearanceGroup() {
        const group = new Adw.PreferencesGroup({title: 'Appearance'});
        const sp = readSpeech();
        const dragged = !!readPillSpot();
        const choices = dragged ? [...PILL_CHOICES, ['dragged', 'Where You Dragged It']] : PILL_CHOICES;
        group.add(choiceRow({title: 'Indicator Position',
            subtitle: 'Where the recording indicator appears. You can also drag it with the mouse'},
        choices, dragged ? 'dragged' : sp.pill, v => {
            if (v === 'dragged')
                return;
            writePillSpot(null);
            writeSpeech({...readSpeech(), pill: v});
        }));
        return group;
    }

    // footer: who made it
    _aboutGroup() {
        const group = new Adw.PreferencesGroup();
        const box = new Gtk.Box({orientation: Gtk.Orientation.VERTICAL, spacing: 6, margin_top: 12,
            halign: Gtk.Align.CENTER});
        box.append(new Gtk.Label({label: 'Proudly made in Ukraine 🇺🇦', css_classes: ['heading']}));
        box.append(new Gtk.LinkButton({uri: REPO, label: 'Project Page', halign: Gtk.Align.CENTER}));
        group.add(box);
        return group;
    }

    _speechPage(window) {
        const page = new Adw.PreferencesPage({title: 'Speech', icon_name: 'audio-input-microphone-symbolic'});
        const sp = readSpeech();
        // the indicator position and the model are saved into the same file elsewhere: keep them
        const save = () => {
            const now = readSpeech();
            writeSpeech({...sp, pill: now.pill, model: now.model});
        };
        let show = () => {};        // shows the languages of the chosen model (set below)
        const [model, packs, ready, modelNow] = this._modelGroup(window, info => show(info));
        page.add(model);
        if (packs)
            page.add(packs);

        // languages of the chosen model: the main one first, then the others in the table's order
        const langs = new Adw.PreferencesGroup({title: 'Languages',
            description: 'Only the languages you turn on are recognised, so similar ones are not mixed up. ' +
                'Changes apply from the next recording.' +
                (modelNow()?.small ? ' The list shows the languages of the chosen model.' : '')});
        const names = new Map(LANGUAGES.map(([c, en, own]) => [c, en === own ? en : `${en} (${own})`]));
        let small = null;           // the helper's model info while the small model is chosen, else null
        let codes = LARGE_CODES;    // the chosen model's languages
        const enabled = () => sp.languages.filter(c => codes.has(c));
        // [name, pack] of the language pack to download for a language no installed pack has (small model)
        const missing = code => {
            const all = Object.entries(small?.small.packs ?? {});
            return all.some(([, p]) => p.installed && p.languages?.includes(code)) ? null
                : all.find(([, p]) => p.languages?.includes(code)) ?? null;
        };
        const main = new Adw.ComboRow({title: 'Main Language'});
        const fillMain = () => {
            main.model = Gtk.StringList.new(enabled().map(c => names.get(c)));
            main.selected = 0;
        };
        main.connect('notify::selected', () => {
            const c = enabled()[main.selected];
            if (main.selected > 0 && c) {
                sp.languages = [c, ...sp.languages.filter(x => x !== c)];
                save();
                fillMain();
            }
        });
        langs.add(main);
        const list = new Adw.ExpanderRow({title: 'Enabled Languages'});
        const switches = new Map();
        const summary = () => {
            const on = enabled();
            list.subtitle = on.length ? on.map(c => NAMES.get(c)).join(', ') : 'None, turn one on';
            for (const [code, row] of switches) {
                const pack = row.active && missing(code);
                row.subtitle = pack ? esc(`Needs the language pack “${inWords(pack[1].languages)}” above`) : '';
            }
        };
        for (const [code] of LANGUAGES) {
            const row = new Adw.SwitchRow({title: names.get(code), active: sp.languages.includes(code)});
            row.connect('notify::active', () => {
                if (row.active && !sp.languages.includes(code)) {
                    sp.languages.push(code);
                    const pack = missing(code);
                    if (pack) {
                        toast(window, `${NAMES.get(code)} needs the language pack ` +
                            `“${inWords(pack[1].languages)}”: download it above`);
                    }
                } else if (!row.active && sp.languages.includes(code)) {
                    if (enabled().length === 1) {        // at least one language stays on
                        row.active = true;
                        return;
                    }
                    sp.languages = sp.languages.filter(x => x !== code);
                }
                save();
                fillMain();
                summary();
            });
            list.add_row(row);
            switches.set(code, row);
        }
        langs.add(list);
        page.add(langs);
        show = info => {
            small = info?.small && current(info) === 'small' ? info : null;
            codes = small ? smallCodes(small) : LARGE_CODES;
            for (const [code, row] of switches)
                row.visible = codes.has(code);
            main.subtitle = small ? 'Preferred when similar languages are hard to tell apart'
                : 'Preferred when Ukrainian, Russian and Bulgarian are hard to tell apart';
            fillMain();
            summary();
        };
        show(modelNow());

        // text
        const text = new Adw.PreferencesGroup({title: 'Text'});
        const space = new Adw.SwitchRow({title: 'Space After the Text',
            subtitle: 'So that the next dictation does not stick to this one', active: sp.spaceAfter});
        space.connect('notify::active', () => {
            sp.spaceAfter = space.active;
            save();
        });
        const period = choiceRow({title: 'No Period at the End', subtitle: 'Chats read better without it'},
            [['keep', 'Never'], ['chats', 'In Chats and Terminals'], ['drop', 'Always']], sp.finalPeriod, v => {
                sp.finalPeriod = v;
                save();
            });
        const punct = new Adw.SwitchRow({title: 'Spoken Punctuation',
            subtitle: 'Say “кома”, “крапка”, “знак питання”, “новий рядок”, “comma”, “new line”… ' +
                '(Ukrainian, Russian, English). A new line is a space in a terminal',
            active: sp.spokenPunctuation});
        punct.connect('notify::active', () => {
            sp.spokenPunctuation = punct.active;
            save();
        });
        const fillers = new Adw.SwitchRow({title: 'Remove Hesitation Sounds',
            subtitle: '“еее”, “ммм”, “um”, “uh”', active: sp.fillers});
        fillers.connect('notify::active', () => {
            sp.fillers = fillers.active;
            save();
        });
        const repl = new Adw.ActionRow({title: 'Replacements and Snippets',
            subtitle: 'Fix words (“пайтон = Python”) or expand phrases (“мій імейл = me@example.com”)'});
        const edit = new Gtk.Button({label: 'Edit', valign: Gtk.Align.CENTER});
        edit.connect('clicked', () => {
            const file = Gio.File.new_for_path(ensureReplacements());
            Gio.AppInfo.launch_default_for_uri_async(file.get_uri(), null, null, (o, res) => {
                try {
                    Gio.AppInfo.launch_default_for_uri_finish(res);
                } catch (e) {
                    repl.subtitle = esc(`Open it by hand: ${file.get_path()}`);
                }
            });
        });
        repl.add_suffix(edit);
        repl.activatable_widget = edit;
        for (const r of [punct, fillers, period, space, repl])
            text.add(r);
        page.add(text);

        // recording
        const rec = new Adw.PreferencesGroup({title: 'Recording'});
        const mics = [['default', 'System Default'], ...microphones().map(([n, d]) =>
            [n, d.replace(/ \(HD Audio\)| Family On-Package High Definition Audio/g, '')])];
        if (sp.microphone !== 'default' && !mics.some(([n]) => n === sp.microphone))
            mics.push([sp.microphone, 'Chosen microphone (not connected now)']);
        const mic = choiceRow({title: 'Microphone',
            subtitle: 'If the chosen one is not connected, the system default is used'},
        mics, sp.microphone, v => {
            sp.microphone = v;
            save();
        });
        rec.add(mic);
        rec.add(choiceRow({title: 'Stop on Silence', subtitle: 'Finish by itself when you stop talking'},
            STOP_CHOICES, sp.autoStop, v => {
                sp.autoStop = v;
                save();
            }));
        rec.add(choiceRow({title: 'Ignore Quieter Voices',
            subtitle: 'Mute sound much quieter than your voice: other people, a TV. Stronger works best with a headset'},
        GATE_CHOICES, sp.voiceGate, v => {
            sp.voiceGate = v;
            save();
        }));
        page.add(rec);

        // performance
        const perf = new Adw.PreferencesGroup({title: 'Performance'});
        const threads = new Adw.ComboRow({title: 'Processor Threads',
            subtitle: `More is not always faster; automatic uses up to 4 of ${GLib.get_num_processors()}`,
            model: Gtk.StringList.new(THREAD_CHOICES.map(([, l]) => l))});
        const ti = THREAD_CHOICES.findIndex(([n]) => n === sp.threads);
        if (ti >= 0) {
            threads.selected = ti;
        } else {
            threads.model.append(String(sp.threads));
            threads.selected = THREAD_CHOICES.length;
        }
        threads.connect('notify::selected', () => {
            sp.threads = THREAD_CHOICES[threads.selected]?.[0] ?? sp.threads;
            save();
        });
        perf.add(threads);
        page.add(perf);
        return [page, ready];
    }

    // The model: Small (tinyasr, with its language packs) or Large (Parakeet, one row per variant): use it,
    // download, remove. Downloading (and converting) is done by the helper (dictate --download), which reports
    // its progress line by line; closing the window stops it (the next download continues where it stopped).
    // A helper without the small model gets the Large rows only, as before. onChange(info) runs after each
    // refresh. Returns [group, the Language Packs group or null, a model is installed, () => model info].
    _modelGroup(window, onChange) {
        let info = modelInfo();
        const group = new Adw.PreferencesGroup({title: 'Speech Model', description: info?.small
            ? 'It runs on this computer, without network access. Small: tinyasr (GPL-3.0-or-later), language ' +
                'packs downloaded from GitHub. Large: NVIDIA Parakeet TDT 0.6B v3 (CC BY 4.0), downloaded from Hugging Face.'
            : 'NVIDIA Parakeet TDT 0.6B v3 (CC BY 4.0), downloaded from Hugging Face. ' +
                'It runs on this computer, without network access.'});
        if (!info) {
            const row = new Adw.ActionRow({title: 'The Speech Engine Is Not Installed',
                subtitle: 'Dictation needs the Dictator helper program, installed separately'});
            row.add_suffix(new Gtk.LinkButton({label: 'How to Install', uri: `${REPO}#install`,
                valign: Gtk.Align.CENTER}));
            group.add(row);
            // the command itself, to select and copy (the extension cannot install a program itself)
            group.add(new Adw.ActionRow({title: 'In a Terminal', subtitle_selectable: true,
                subtitle: esc(`git clone ${REPO}.git\ncd dictator && ./install.sh --no-extension`)}));
            return [group, null, false, () => null];
        }
        let trial = null;           // the Try Dictation row, shown once a model is installed
        const rows = new Map();     // what --download takes ('int8', 'small', 'small:en-fa'...) -> its row
        let job = null;             // {id, proc, pulse, cancelled}
        let updating = false;
        let large = null, packs = null;     // with the small model: the Large row, the Language Packs group

        const refresh = () => {
            info = modelInfo() ?? info;
            const busy = !!job || !!info.busy;
            updating = true;
            for (const [id, u] of rows) {
                const running = job?.id === id;
                const installed = u.installed();
                if (u.radio) {
                    u.radio.sensitive = installed;
                    u.radio.active = info.used === id;
                }
                u.download.visible = !installed && !running && (u.canDownload?.() ?? true);
                u.remove.visible = installed && !running && !!u.ask;
                u.download.sensitive = u.remove.sensitive = !busy;
                u.bar.visible = u.cancel.visible = running;
                if (!running)
                    u.row.subtitle = esc(u.text());
            }
            updating = false;
            if (large) {
                const v = VARIANTS.find(x => x.id === info.used);
                large.subtitle = `NVIDIA Parakeet, ${LARGE_CODES.size} languages${v ? `. In use: ${v.title}` : ''}`;
            }
            if (packs)
                packs.visible = current(info) === 'small' || !!info.small.installed || !!job?.id.startsWith('small');
            trial?.row.set_visible(!!info.used && !job);
            onChange(info);
        };

        const stopJob = () => {
            if (job?.pulse)
                GLib.source_remove(job.pulse);
            if (job)
                job.pulse = 0;
        };

        const download = id => {
            const u = rows.get(id);
            let proc;
            try {
                proc = Gio.Subprocess.new([HELPER, '--download', id],
                    Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_SILENCE);
            } catch (e) {
                toast(window, `Cannot start the download: ${e.message}`);
                return;
            }
            job = {id, proc, pulse: 0, cancelled: false};
            u.bar.fraction = 0;
            u.row.subtitle = 'Starting…';
            refresh();
            let error = null;
            const stream = new Gio.DataInputStream({base_stream: proc.get_stdout_pipe()});
            const read = () => stream.read_line_async(GLib.PRIORITY_DEFAULT, null, (st, res) => {
                let line = null;
                try {
                    [line] = st.read_line_finish_utf8(res);
                } catch (e) {}
                if (line === null)
                    return;
                const [kind, stage, done, total] = line.split(' ');
                if (kind === 'E') {
                    error = line.slice(2);
                } else if (kind === 'P' && stage === 'download') {
                    u.bar.fraction = Number(done) / Number(total);
                    u.row.subtitle = `Downloading… ${u.size(Number(done))} of ${u.size(Number(total))}. ` +
                        'Keep this window open';
                } else if (kind === 'P' && done === '0' && !job.pulse) {
                    u.row.subtitle = stage === 'convert' ? 'Converting to 8-bit on this computer…'
                        : 'Setting it up…';
                    job.pulse = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 150, () => {
                        u.bar.pulse();
                        return GLib.SOURCE_CONTINUE;
                    });
                }
                read();
            });
            read();
            proc.wait_async(null, (p, res) => {
                try {
                    p.wait_finish(res);
                } catch (e) {}
                stopJob();
                const cancelled = job.cancelled;
                job = null;
                if (!cancelled)
                    toast(window, error ?? (p.get_successful() ? u.done : 'The download failed'));
                refresh();
            });
        };

        const remove = id => {
            const [heading, body] = rows.get(id).ask();
            const dlg = new Adw.AlertDialog({heading, body});
            dlg.add_response('cancel', 'Cancel');
            dlg.add_response('remove', 'Remove');
            dlg.set_response_appearance('remove', Adw.ResponseAppearance.DESTRUCTIVE);
            dlg.connect('response', (d, response) => {
                if (response !== 'remove')
                    return;
                try {
                    Gio.Subprocess.new([HELPER, '--remove', id], Gio.SubprocessFlags.STDERR_SILENCE).wait(null);
                } catch (e) {
                    toast(window, `Cannot remove it: ${e.message}`);
                }
                refresh();
            });
            dlg.present(window);
        };

        // one row: a radio button to use it (spec.choose), Download, Remove (spec.ask), progress and Cancel;
        // spec: {installed(), text(), size(bytes), done (the toast), ask() -> [heading, body], canDownload()}
        let first = null;
        const addRow = (put, id, title, spec) => {
            const row = new Adw.ActionRow({title: esc(title)});
            let radio = null;
            if (spec.choose) {
                radio = new Gtk.CheckButton({valign: Gtk.Align.CENTER, group: first});
                first ??= radio;
                radio.connect('toggled', () => {
                    if (!updating && radio.active)
                        writeSpeech({...readSpeech(), model: id});
                    if (!updating)
                        refresh();
                });
                row.add_prefix(radio);
                row.activatable_widget = radio;
            }
            const dl = new Gtk.Button({label: 'Download', valign: Gtk.Align.CENTER});
            dl.connect('clicked', () => download(id));
            const rm = new Gtk.Button({icon_name: 'user-trash-symbolic', valign: Gtk.Align.CENTER,
                tooltip_text: 'Remove', css_classes: ['flat']});
            rm.connect('clicked', () => remove(id));
            const bar = new Gtk.ProgressBar({valign: Gtk.Align.CENTER, width_request: 120});
            const cancel = new Gtk.Button({label: 'Cancel', valign: Gtk.Align.CENTER});
            cancel.connect('clicked', () => {
                job.cancelled = true;
                job.proc.send_signal(15);       // SIGTERM: the helper stops its download or conversion
            });
            for (const w of [bar, cancel, dl, rm])
                row.add_suffix(w);
            put(row);
            rows.set(id, {...spec, row, radio, download: dl, remove: rm, bar, cancel});
        };

        if (info.small) {
            const packDisk = () => Object.values(info.small.packs ?? {}).reduce((n, p) => n + (p.disk ?? 0), 0);
            addRow(row => group.add(row), 'small', 'Small', {choose: true, size: mb, done: 'The small model is ready',
                installed: () => !!info.small.installed, canDownload: () => !!info.small.worker,
                text: () => ['Recommended. Fast, a 30 to 33 MB download per language pack',
                    !info.small.worker ? 'Its engine is not built on this computer yet: run ./install.sh again'
                        : info.small.installed ? `${info.used === 'small' ? 'In use' : 'Installed'}, ${mb(packDisk())}`
                            : null].filter(Boolean).join('\n')});
            large = new Adw.ExpanderRow({title: 'Large', expanded: VARIANTS.some(v => v.id === current(info))});
            group.add(large);
        }
        for (const v of VARIANTS) {
            addRow(row => (large ? large.add_row(row) : group.add(row)), v.id, v.title, {choose: true, size: gb,
                done: 'The speech model is ready', installed: () => info.variants[v.id].installed,
                text: () => {
                    const x = info.variants[v.id];
                    return x.installed ? `${info.used === v.id ? 'In use' : 'Installed'}, ${gb(x.disk)}` : v.note(x);
                },
                ask: () => [`Remove the ${v.title} Model?`,
                    `This frees ${gb(info.variants[v.id].disk)}. You can download it again later.`]});
        }
        trial = this._trialRow(window);
        group.add(trial.row);
        if (info.small) {
            packs = new Adw.PreferencesGroup({title: 'Language Packs',
                description: 'For the small model: download the ones with your languages.'});
            for (const name of Object.keys(info.small.packs ?? {})) {
                const pack = () => info.small.packs?.[name] ?? {};
                addRow(row => packs.add(row), `small:${name}`, inWords(pack().languages ?? [name]), {size: mb,
                    done: 'The language pack is ready', installed: () => !!pack().installed,
                    text: () => (pack().installed ? `Installed, ${mb(pack().disk)}`
                        : `${mb(pack().download)} download, takes ${mb(pack().size)}`),
                    ask: () => ['Remove This Language Pack?', `${inWords(pack().languages ?? [name])}. ` +
                        `This frees ${mb(pack().disk)}. You can download it again later.`]});
            }
        }
        window.connect('close-request', () => {
            if (job) {
                job.cancelled = true;
                job.proc.send_signal(15);
            }
            stopJob();
            trial.cancel();
            return false;
        });
        refresh();
        if (info.busy)
            group.description += '\nA download is running in another window.';
        return [group, packs, !!info.used, () => info];
    }

    // A test dictation (dictate --try): the voice level while you speak, then the text, here and
    // nowhere else; nothing is kept. Returns {row, cancel()}.
    _trialRow(window) {
        const row = new Adw.ActionRow({title: 'Try Dictation',
            subtitle: 'Say a sentence: the text appears here and is typed nowhere'});
        const level = new Gtk.LevelBar({valign: Gtk.Align.CENTER, width_request: 96, visible: false});
        const button = new Gtk.Button({label: 'Try', valign: Gtk.Align.CENTER});
        row.add_suffix(level);
        row.add_suffix(button);
        let proc = null, stdin = null;
        const send = cmd => {
            try {
                stdin?.write_all(new TextEncoder().encode(`${cmd}\n`), null);
                stdin?.flush(null);
            } catch (e) {}
        };
        button.connect('clicked', () => {
            if (proc) {                 // the second click: finish
                send('stop');
                button.sensitive = false;
                return;
            }
            try {
                proc = Gio.Subprocess.new([HELPER, '--try'], Gio.SubprocessFlags.STDIN_PIPE |
                    Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_SILENCE);
            } catch (e) {
                row.subtitle = esc(`Cannot start: ${e.message}`);
                return;
            }
            const mine = proc;
            stdin = proc.get_stdin_pipe();
            button.label = 'Stop';
            level.value = 0;
            level.visible = true;
            row.subtitle = 'Listening… speak, then press Stop';
            const stream = new Gio.DataInputStream({base_stream: proc.get_stdout_pipe()});
            const read = () => stream.read_line_async(GLib.PRIORITY_DEFAULT, null, (st, res) => {
                let line = null;
                try {
                    [line] = st.read_line_finish_utf8(res);
                } catch (e) {}
                if (line === null || proc !== mine)
                    return;
                if (line.startsWith('L ')) {
                    level.value = Math.max(0, Math.min(1, parseFloat(line.slice(2)) || 0));
                } else if (line === 'S') {
                    level.visible = false;
                    row.subtitle = 'Transcribing…';
                } else if (line.startsWith('T ')) {
                    let text = '';
                    try {
                        text = JSON.parse(line.slice(2)).replaceAll(NEWLINE_MARK, ' ');
                    } catch (e) {}
                    row.subtitle = esc(text ? `“${text}”` : 'No speech recognised');
                } else if (line.startsWith('E ')) {
                    row.subtitle = esc(line.slice(2));
                }
                read();
            });
            read();
            proc.wait_async(null, () => {
                if (proc !== mine)
                    return;
                proc = stdin = null;
                level.visible = false;
                button.label = 'Try';
                button.sensitive = true;
            });
        });
        const cancel = () => {
            if (!proc)
                return;
            send('cancel');             // the helper stops the microphone and the model, and exits
            try {
                stdin?.close(null);
            } catch (e) {}
            proc = stdin = null;
        };
        window.connect('unmap', cancel);
        return {row, cancel};
    }

    _privacyGroup() {
        const group = new Adw.PreferencesGroup({
            title: 'Privacy',
            description: 'Text is typed straight into the app, without the clipboard. Recordings stay in ' +
                'memory only, never on the disk, and are deleted after the time below.',
        });
        const p = readPrivacy();
        const history = new Adw.SwitchRow({title: 'Recent Texts in the Top Bar',
            subtitle: 'The last 3 texts in the microphone menu, to copy them',
            active: p.history});
        const clipboard = new Adw.SwitchRow({title: 'Paste Through the Clipboard',
            subtitle: 'For apps that cannot take typed text. Off: the text waits in the microphone menu, ' +
                'and other apps never see it in the clipboard',
            active: p.clipboard});
        const keep = new Adw.ComboRow({title: 'Delete Recordings After',
            model: Gtk.StringList.new(KEEP_CHOICES.map(([, label]) => label))});
        const idx = KEEP_CHOICES.findIndex(([m]) => m === p.keepMinutes);
        if (idx >= 0) {
            keep.selected = idx;
        } else {                    // a value set by hand in the file: show it as it is
            keep.model.append(`${p.keepMinutes} minutes`);
            keep.selected = KEEP_CHOICES.length;
        }
        const audio = new Adw.SwitchRow({title: 'Keep Audio to Transcribe Again',
            subtitle: 'The recordings of those texts stay in memory too, so the menu can transcribe them ' +
                'again, also in another language',
            active: p.keepAudio});
        // the languages the menu offers; until you change them, the enabled ones (saved only once changed)
        const again = new Adw.ExpanderRow({title: 'Transcribe Again In'});
        let chosen = againLanguages(p), touched = p.again !== null;
        const summary = () => {
            again.subtitle = chosen.length ? LANGUAGES.filter(([c]) => chosen.includes(c)).map(([, en]) => en)
                .join(', ') : 'Only as it was';
        };
        for (const [code, en, own] of LANGUAGES) {
            const row = new Adw.SwitchRow({title: en === own ? en : `${en} (${own})`, active: chosen.includes(code)});
            row.connect('notify::active', () => {
                chosen = row.active ? [...new Set([...chosen, code])] : chosen.filter(c => c !== code);
                touched = true;
                summary();
                save();
            });
            again.add_row(row);
        }
        summary();
        const sync = () => {
            audio.sensitive = history.active;
            again.sensitive = history.active && audio.active;
        };
        const save = () => {
            const minutes = KEEP_CHOICES[keep.selected]?.[0] ?? readPrivacy().keepMinutes;
            writePrivacy({history: history.active, keepMinutes: minutes, clipboard: clipboard.active,
                keepAudio: audio.active, again: touched ? chosen : null});
            sync();
        };
        for (const w of [history, clipboard, audio])
            w.connect('notify::active', save);
        keep.connect('notify::selected', save);
        sync();
        for (const w of [clipboard, history, audio, again, keep])
            group.add(w);
        return group;
    }

    // a small window that waits for the new shortcut
    _capture(parent, r, set) {
        const extra = [r.canOff ? 'Backspace: off' : null,
            r.canSame ? 'Backspace or the start shortcut: the start shortcut again' : null,
            r.caps ? 'Caps Lock: use Caps Lock' : null, r.shiftCaps ? 'Shift+Caps Lock works too' : null]
            .filter(Boolean).join(', ');
        const hint = `Esc: keep the old one${extra ? `, ${extra}` : ''}`;
        const dlg = new Adw.Window({transient_for: parent, modal: true, default_width: 460, default_height: 280,
            title: r.title});
        const status = new Adw.StatusPage({icon_name: 'input-keyboard-symbolic',
            title: `Press the new shortcut for “${r.title}”`, description: hint});
        const box = new Gtk.Box({orientation: Gtk.Orientation.VERTICAL});
        box.append(new Adw.HeaderBar());
        box.append(status);
        dlg.set_content(box);
        // A key can come with a twin: a keyboard tool that turns Shift+Caps Lock into another key sends
        // both. So the window takes keys a moment longer before it closes, and nothing reaches GNOME.
        let done = false;
        const finish = value => {
            if (done)
                return Gdk.EVENT_STOP;
            const err = set(r.id, value);
            if (err) {
                status.description = esc(`${err}\n${hint}`);
                return Gdk.EVENT_STOP;
            }
            done = true;
            status.title = `${r.title}: ${value === SAME ? `${prettyAccel(readKeys().start[0])} again`
                : prettyAccel(value)}`;
            status.description = '';
            GLib.timeout_add(GLib.PRIORITY_DEFAULT, CLOSE_MS, () => {
                dlg.close();
                return GLib.SOURCE_REMOVE;
            });
            return Gdk.EVENT_STOP;
        };
        const refuse = msg => {
            status.description = esc(`${msg}\n${hint}`);
            return Gdk.EVENT_STOP;
        };
        // capture phase: Enter / Space must reach this, not activate the window's close button
        const ctl = new Gtk.EventControllerKey({propagation_phase: Gtk.PropagationPhase.CAPTURE});
        ctl.connect('key-pressed', (c, keyval, keycode, state) => {
            if (done)
                return Gdk.EVENT_STOP;
            const mods = state & Gtk.accelerator_get_default_mod_mask() & ~Gdk.ModifierType.LOCK_MASK;
            const name = Gdk.keyval_name(keyval) ?? '';
            if (!mods && keyval === Gdk.KEY_Escape)
                return dlg.close() ?? Gdk.EVENT_STOP;
            if (!mods && keyval === Gdk.KEY_BackSpace && r.canOff)
                return finish(NONE);
            if (!mods && keyval === Gdk.KEY_BackSpace && r.canSame)
                return finish(SAME);
            if (keyval === Gdk.KEY_Caps_Lock && mods === Gdk.ModifierType.SHIFT_MASK && r.shiftCaps)
                return finish(SHIFT_CAPS);
            if (keyval === Gdk.KEY_Caps_Lock) {
                return r.caps ? finish(CAPS) : refuse(r.shiftCaps ? 'Caps Lock works here only with Shift.'
                    : 'Caps Lock works only for Pause.');
            }
            if (MODIFIER_KEY.test(name))
                return Gdk.EVENT_STOP;          // wait for the whole combination
            // the key without Shift applied: Shift+1 is <Shift>1, not <Shift>exclam
            let key = keyval;
            const [ok, base] = dlg.get_display().translate_key(keycode, 0, c.get_group());
            if (ok && base)
                key = base;
            key = Gdk.keyval_to_lower(key);
            if (key === Gdk.KEY_KP_Enter)       // both Enter keys work as "Return"
                key = Gdk.KEY_Return;
            const strong = mods & (Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.ALT_MASK |
                Gdk.ModifierType.SUPER_MASK | Gdk.ModifierType.META_MASK | Gdk.ModifierType.HYPER_MASK);
            if (r.id === 'start' && !strong && !BARE_OK.test(Gdk.keyval_name(key) ?? ''))
                return refuse('Add Ctrl, Alt or Super: on its own this key would stop typing everywhere.');
            if (!Gtk.accelerator_valid(key, mods))
                return refuse('That is not a valid shortcut.');
            const accel = Gtk.accelerator_name(key, mods);
            if (r.canSame && readKeys().start.some(a => same(a, accel)))    // finish = start: one key toggles
                return finish(SAME);
            return finish(accel);
        });
        dlg.add_controller(ctl);
        // let the combination reach this window, not GNOME (e.g. the current dictation shortcut)
        dlg.connect('map', () => dlg.get_surface()?.inhibit_system_shortcuts?.(null));
        dlg.connect('unmap', () => dlg.get_surface()?.restore_system_shortcuts?.());
        dlg.present();
    }
}
