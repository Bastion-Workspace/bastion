---
title: Map and locations overview
order: 1
---

# Map and locations overview

The **Map** page shows an **interactive map** with **locations** you add. You can create, edit, and delete locations, view them on the map, and get **driving directions** (routing) between two points. Map **style** (light, auto, or dark) can be changed. This page describes how to open the Map, manage locations, and use routing. The Map entry may be **capability-gated** (e.g. only admins or users with a maps feature).

---

## Opening the Map

- **Navigation** — Click **Map** in the main navigation. The Map page opens with a **map view** and a **locations** list or panel. If you do not see Map, your instance may require a capability (e.g. `feature.maps.view`) or admin role; ask your administrator.
- **URL** — The Map page is at `/map`.

---

## Locations

- **Adding a location** — Use **Add** or **New location** (e.g. from the list or a toolbar). In the dialog, enter **name**, **address** or **coordinates**, and optional **notes**. Save. The location appears in the list and as a **marker** on the map.
- **Editing** — Open a location from the list (or click its marker) and choose **Edit**. Change name, address, or notes and save.
- **Deleting** — Select a location and choose **Delete**. Confirm. The location is removed from the list and the map.
- **List view** — The **Locations** list shows all your locations. Click one to focus it on the map or to edit. You can **search** or filter if the UI supports it.

---

## Map view

- **Interactive map** — The main area is a **map** (e.g. OpenStreetMap or another provider). You can **pan** and **zoom**. **Markers** show your locations; clicking a marker may select the location and show a popup or details.
- **Map style** — Switch the map **style** (e.g. **light**, **auto**, **dark**) from a control or settings. “Auto” may follow your app theme (light/dark). The choice is often saved in local storage so it persists.

---

## Routing (driving directions)

- **Getting directions** — Select **two locations** (e.g. from the list or by clicking markers) and choose **Route** or **Get directions** (e.g. **DirectionsCar** or “Driving” icon). The map draws the **route** and may show **turn-by-turn** steps (e.g. “Turn left onto Main St”, “Arrive at destination”) and **distance** (e.g. in miles or feet). Routing uses the configured provider (e.g. OSRM or similar).
- **Steps** — The **steps** or **legs** of the route are listed; you can click a step to highlight it on the map. **Play** or **Start** may begin turn-by-turn guidance if the instance supports it.

---

## Capability gating

- **Visibility** — The **Map** nav entry is only visible to users who have the maps capability (e.g. admin or `feature.maps.view`). If you do not see Map, you do not have access.
- **Permissions** — Locations may be per-user or shared depending on instance configuration. Editing and deleting are typically limited to the owner or admins.

---

## Summary

- **Map** is in the main nav (capability-gated). The page has an **interactive map** and a **locations** list. **Add**, **edit**, and **delete** locations; they appear as **markers** on the map.
- **Routing** gives **driving directions** between two locations with **turn-by-turn** steps and **distance**. **Map style** (light/auto/dark) can be changed.

---

## Related

- **Welcome** — Main areas and navigation.
- **Status bar** — Date/time and weather (location can affect weather).
- **Settings overview** — User profile (e.g. timezone) and other options.
