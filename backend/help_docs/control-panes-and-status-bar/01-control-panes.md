---
title: Control panes
order: 1
---

# Control panes

**Control panes** are custom UI controls (sliders, dropdowns, toggles, buttons, text displays) that are bound to **external endpoints**. You define panes and their type; they appear in the **status bar** as icons or in a dedicated test panel. They are used to drive smart-home or automation systems, or any service that accepts HTTP or similar calls. This page describes what control panes are, how to create and configure them, and how they appear in the status bar.

---

## What control panes are

A **control pane** is a single control that sends a value or action to an external system:

- **Slider** — A numeric value (e.g. 0–100) sent to an endpoint when the user moves the slider.
- **Dropdown** — A selected option sent when the user picks from a list.
- **Toggle** — On/off (or two-state) value sent when the user toggles.
- **Button** — A single action: when pressed, the pane calls an endpoint (e.g. trigger a scene or command).
- **Text display** — Read-only text (e.g. status or response from an endpoint). May be updated by polling or by a webhook.

Each pane has a **name**, **type**, and **endpoint** (URL and optional method/body). The backend or a proxy calls that endpoint when the user interacts with the control. Control panes are configured on the **Control Panes** page (reachable from the user menu or **Settings**).

---

## Opening the Control Panes page

- **User menu** — In the top bar, open the user menu and choose **Control Panes** (or **Control pane configuration**). You may also be redirected from **Settings** when you open the “Control panes” tab (e.g. `/settings?tab=control-panes` may take you to `/control-panes`).
- **URL** — The Control Panes page is at `/control-panes`. The page explains that you can configure status bar controls and pane behavior.

---

## Creating and configuring panes

- **Add pane** — On the Control Panes page, use **Add** or **New control** (the exact label depends on the UI). You then set:
  - **Name** — A short label (e.g. “Living room dimmer”) so you can recognize it in the status bar and test panel.
  - **Type** — Slider, Dropdown, Toggle, Button, or Text display.
  - **Endpoint** — URL (and optionally HTTP method, headers, body template). For a slider, the value might be sent as a query parameter or in the body; for a button, a POST may be sent with no body. The backend or proxy that receives the request is your own service (e.g. Home Assistant, custom API).
  - **Options** (for dropdown) — List of options and the value sent for each. For toggle, the on/off values.
  - **Range** (for slider) — Min, max, and step.

- **Save** — After saving, the pane is registered. It can appear in the **status bar** (as an icon or small control) and in the **Control test panel** (if available) so you can verify behavior without leaving the page.

---

## Status bar icons

- **Control pane icons** — The **status bar** at the bottom of the app can show **icons** (or small controls) for each configured pane. Clicking an icon may open a popover with the full control (slider, toggle, etc.) or trigger the action (for a button). So you can adjust a dimmer or trigger a scene without opening the Control Panes page.
- **Order and visibility** — The order of panes in the status bar may be configurable (e.g. drag-and-drop on the Control Panes page). You can disable or remove a pane so its icon no longer appears.

---

## Testing controls

- **Control test panel** — On the Control Panes page there may be a **test** section or **test panel** that lets you trigger each control and see the request (e.g. URL and payload) or response. Use it to confirm the endpoint is called correctly before relying on the status bar. If your endpoint is not reachable from the browser (e.g. local network), the test may run via the backend proxy.

---

## Summary

- **Control panes** are custom controls (slider, dropdown, toggle, button, text display) bound to **external endpoints**. Configure them on the **Control Panes** page (user menu or `/control-panes`).
- Each pane has **name**, **type**, and **endpoint** (URL and optional method/body). **Status bar** shows **icons** for panes; click to use the control. Use the **test panel** to verify endpoint calls.

---

## Related

- **Status bar** — Where control pane icons and other status bar items appear.
- **Settings overview** — User menu and navigation.
- **External connections** — Email and messaging (different from control pane endpoints).
