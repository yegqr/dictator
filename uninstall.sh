#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025-2026 The Dictator authors (see AUTHORS)
# Removes Dictator.   ./uninstall.sh        keeps the speech models and your settings
#                     ./uninstall.sh --all  also deletes the models (Small and Large), the settings and
#                                           Dictator's container images (the Fedora base image stays)
set -uo pipefail
UUID=dictator@kcharukovskyi.github.io
if command -v gsettings >/dev/null; then
  cur=$(gsettings get org.gnome.shell enabled-extensions 2>/dev/null) &&
  gsettings set org.gnome.shell enabled-extensions "$(/usr/bin/python3 -I -c 'import ast, sys
l = ast.literal_eval(sys.argv[1].replace("@as ", ""))
print(repr([x for x in l if x != sys.argv[2]]))' "$cur" "$UUID")"
fi
# ~/.local/lib/dictate: parakeet.py and dictate-tinyasr (the Small model's program)
rm -rf "$HOME/.local/share/gnome-shell/extensions/$UUID" "$HOME/.local/lib/dictate"
rm -f "$HOME/.local/bin/dictate"
rm -rf "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/dictate" "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/dictate-history"
if [ "${1:-}" = "--all" ]; then
  # ~/.local/share/dictate: tinyasr/ (Small) and parakeet-tdt-0.6b-v3/ (Large)
  rm -rf "$HOME/.local/share/dictate" "$HOME/.config/dictate" "$HOME/.cache/dictate"
  for e in podman docker; do
    command -v $e >/dev/null && $e rmi dictate-asr dictate-tinyasr-build >/dev/null 2>&1
  done
  echo "Removed everything, except the Fedora base image of the container builds, if one was pulled."
else
  echo "Removed. Kept: the models (~/.local/share/dictate) and settings (~/.config/dictate); ./uninstall.sh --all deletes them."
fi
echo "Log out and in to unload the extension."
