---
title: Status bar
order: 2
---

# Status bar

The **status bar** is a fixed bar at the **bottom** of the app. It shows the **date and time** (in your timezone and format), **weather** (location, temperature, conditions, moon phase), **music controls** when media is playing, **control pane icons** for custom controls, and the **app version**. This page describes each part and how it uses your settings.

---

## Date and time

- **Display** — The left side of the status bar shows **date** and **time** (e.g. “MM/DD/YYYY - HH:MM:SS”). The time updates every second.
- **Timezone and format** — The **timezone** and **time format** (12-hour vs 24-hour) come from **Settings > User Profile**. Set your **timezone** (e.g. “America/New_York”) and **time format** (12h or 24h) there; the status bar reflects them. If not set, the app uses the browser or a default.

---

## Weather

- **Display** — When weather is available, the status bar shows **location**, **temperature** (e.g. °F), **conditions** (e.g. “Partly cloudy”), and a **moon phase** icon. Hovering the moon icon may show the phase name (e.g. “Full moon”). Weather is loaded from the backend (which may use your **zip** or **location** from User Profile or a system default).
- **Refresh** — Weather data is refreshed periodically (e.g. every 10 minutes). You do not need to refresh manually. If weather does not appear, check that the backend has a location or zip configured and that the weather service is enabled.

---

## Music controls

- **When media is playing** — The **center** (or right) of the status bar shows **music controls**: play/pause, track title, and possibly skip next/previous. These appear when you have started playback from the **Media** page (or from a media source). You can control playback without switching back to Media.
- **When nothing is playing** — The music area may be empty or show a placeholder. Configure **Settings > Media** (e.g. Navidrome, Audiobookshelf) to use media and see controls when playing.

---

## Control pane icons

- **Custom controls** — If you have **control panes** configured (see **Control panes**), their **icons** appear in the status bar (typically on the right, before the version). Click an icon to open the control (slider, toggle, button, etc.) or to trigger the action. Control panes are for external endpoints (e.g. smart home); they are configured on the **Control Panes** page.

---

## App version

- **Display** — The **right** end of the status bar shows **v** followed by the **app version** (e.g. “v1.2.3”). This is the same version shown in **Help > About**. Useful to confirm which build you are running when reporting issues or checking after an update.

---

## Summary

- The **status bar** is at the **bottom** of the app. It shows **date and time** (from User Profile timezone and format), **weather** (location, temperature, conditions, moon phase), **music controls** when media is playing, **control pane icons**, and **app version**.
- **User Profile** (timezone, time format, zip/location) and **Media** and **Control panes** settings determine what appears and how it behaves.

---

## Related

- **Control panes** — Custom controls and status bar icons.
- **Media overview** — Music, audiobooks, podcasts and playback.
- **Settings overview** — User Profile (timezone, format, location) and other tabs.
- **Welcome** — Main areas and navigation.
