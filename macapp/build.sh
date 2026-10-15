#!/bin/zsh
# Builds ~/Applications/Vision Lab.app (ad-hoc signed, stable bundle id com.houghtonlabs.visionlab)
set -e; cd "${0:A:h}"
APP="$HOME/Applications/Vision Lab.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O -o "$APP/Contents/MacOS/VisionLab" main.swift
cp Info.plist "$APP/Contents/Info.plist"
if [[ ! -f AppIcon.icns ]]; then
  ../.venv/bin/python make_icon.py && rm -rf AppIcon.iconset; mkdir AppIcon.iconset
  for s in 16 32 128 256 512; do sips -z $s $s icon.png --out AppIcon.iconset/icon_${s}x${s}.png >/dev/null
    sips -z $((s*2)) $((s*2)) icon.png --out AppIcon.iconset/icon_${s}x${s}@2x.png >/dev/null; done
  iconutil -c icns AppIcon.iconset -o AppIcon.icns; rm -rf AppIcon.iconset icon.png
fi
cp AppIcon.icns "$APP/Contents/Resources/"
codesign --force --deep -s - --identifier com.houghtonlabs.visionlab "$APP"
codesign -dv "$APP" 2>&1 | grep -E "Identifier|Signature"
