---
title: Admin user management
order: 4
---

# Admin user management

**User Management** is an admin-only area where you create, edit, and delete **users**, set **roles** and **capabilities**, and change **passwords**. You open it from **Settings > User Management** (the tab is only visible when you are logged in as an **admin**). This page describes the user list, creating and editing users, roles and capabilities, and password changes.

---

## Accessing User Management

- Go to **Settings** (user menu in the top bar) and open the **User Management** tab. If you do not see it, your account does not have the **admin** role. Only admins can create, edit, or delete users and change roles or capabilities.

---

## User list

- The main view shows a **table** (or list) of all users: **username**, **email**, **display name**, **role**, and optionally **capabilities**. You can search or filter if the UI supports it. From each row you can **Edit**, **Change password**, or **Delete** (with confirmation).

---

## Creating a user

- Click **Add User** or **Create User** to open the create dialog.
- **Username** — Required. Used for login and must be unique.
- **Email** — Optional in some instances; used for notifications or recovery.
- **Password** — Set the initial password. The user can change it after first login if the instance allows.
- **Display name** — Optional; shown in the UI instead of or with the username.
- **Role** — Choose **user** or **admin**. Admins can access User Management, Database settings, and other restricted areas; regular users cannot.
- Save. The new user appears in the list and can log in with the username and password you set.

---

## Editing a user

- Click **Edit** on a user row to open the edit dialog. You can change:
  - **Email**, **display name**, **role**. Username may be read-only or editable depending on the instance.
  - **Capabilities** — A set of feature flags (e.g. `feature.games.view`, `feature.news.view`, `feature.maps.view`) that control whether the user sees **Games**, **News**, or **Map** in the navigation. Grant or revoke capabilities as needed. Other capabilities may exist for specific features; the list depends on deployment.
- Save. Changes apply immediately; the user may need to refresh or log in again to see nav or permission changes.

---

## Roles and capabilities

- **Admin** — Full access: User Management, Database tab, capability grants, and all feature areas. Admins always see Games, News, Map, and other capability-gated items unless restricted by policy.
- **User** — Standard access. Which features they see (Games, News, Map, etc.) is controlled by **capabilities**. By default a new user may have no extra capabilities; grant **feature.games.view**, **feature.news.view**, **feature.maps.view**, etc., to show those nav entries and pages.
- Capabilities are stored per user; editing a user lets you add or remove them.

---

## Changing a user’s password

- Click **Change password** (or the key icon) on a user row. In the dialog, enter the **new password** and confirm. Some instances require the **current** (admin) password or the user’s current password for security. Save. The user can log in with the new password on next login.

---

## Deleting a user

- Click **Delete** on a user row. Confirm when prompted. The user is removed from the system; their data may be retained or purged according to instance policy. Deletion is typically permanent. You cannot delete your own user while logged in as that user in some setups.

---

## Summary

- **User Management** (Settings > User Management) is **admin-only**. Use it to **create**, **edit**, and **delete** users and to set **roles** and **capabilities**.
- **Create user**: username, email, password, display name, role. **Edit user**: change email, display name, role, and **capabilities** (e.g. feature.games.view, feature.news.view, feature.maps.view).
- **Change password** and **Delete** are available per user. Roles: **admin** (full access) and **user** (access controlled by capabilities).

---

## Related

- **Settings overview** — All settings tabs.
- **External connections** — Email and messaging (per-user connections).
- **Models and providers** — Per-user model preferences.
