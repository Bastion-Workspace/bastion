---
title: WebDAV sync overview
order: 1
---

# WebDAV sync overview

Bastion can serve your files over **WebDAV** so you can sync and edit them with **mobile apps** (e.g. **beorg**, **Orgzly**) or any WebDAV-capable file manager. There is no in-app WebDAV configuration screen; you use your **Bastion username and password** in the external client and point it at the WebDAV URL. This page describes the URL, credentials, what you can access, and how to set up common clients.

---

## What WebDAV provides

- **WebDAV server** — A WebDAV endpoint is available at a path such as **`/dav/`** on your Bastion instance. The server maps to the **uploads** directory (or the same file store the Document Library uses), so the **folder structure** you see in the app is the same as in WebDAV.
- **Use case** — Sync org files, Markdown, or other documents to your phone or tablet; edit them in beorg, Orgzly, or another app; changes sync back when the app uploads. You can also use a desktop WebDAV client to mount the folder as a drive.

---

## URL and credentials

- **URL** — Use your instance base URL plus **`/dav/`**. For example: **`https://your-instance.example.com/dav/`**. The trailing slash is often required. This is the **root** of the WebDAV tree (uploads root).
- **Username** — Your **Bastion username** (the one you use to log in to the web app).
- **Password** — Your **Bastion password**. Some clients support saving it; use a secure connection (HTTPS) so the password is not sent in the clear.
- **Protocol** — **HTTPS** is required. The server does not serve WebDAV over plain HTTP in normal deployments.

---

## What you can access

- **Files and folders** — You see the same **folders and files** as in the uploads directory. The hierarchy (e.g. `My Documents`, subfolders, files) is preserved. You can **browse**, **download**, **upload**, **edit**, and **delete** according to the same permissions as in the web app (depending on how the server is configured).
- **All file types** — WebDAV serves **all** file types (e.g. `.org`, `.md`, images, PDFs), not only org files. Mobile org apps will typically show org files; file managers show everything.
- **Limitation** — The server may currently expose **all** files in the uploads directory (e.g. not filtered per user in multi-tenant setups). Check your instance documentation or admin for isolation and security details.

---

## Client setup

- **beorg (iOS)** — Add a **WebDAV** account: enter the **WebDAV URL** (e.g. `https://your-instance.example.com/dav/`), **username**, and **password**. beorg will list folders and org files; you can sync and edit.
- **Orgzly (Android)** — Add a **WebDAV** repository: set **URL** to the same `/dav/` address, **username** and **password**. Orgzly will discover org files and sync.
- **Other file managers** — Any app that supports WebDAV (e.g. **FolderSync**, **Solid Explorer**, or desktop **Finder** / **Windows Map network drive**) can use the same URL and credentials to browse and copy files.

---

## No in-app UI

- WebDAV is a **server-side** feature. There is no **Settings > WebDAV** or in-app form to “enable” it; if your instance is deployed with the WebDAV service, the `/dav/` URL is already available. You only configure the **external client** with the URL and your Bastion login.

---

## Summary

- **WebDAV** is available at **`https://<your-instance>/dav/`**. Use your **Bastion username** and **password** in the client. **HTTPS** required.
- You can access the same **files and folders** as in the Document Library; structure is preserved. Use **beorg**, **Orgzly**, or any WebDAV client to sync and edit. There is **no in-app** WebDAV configuration.

---

## Related

- **Document Library overview** — How files and folders are organized.
- **Org Mode files** — Org files you might sync via WebDAV.
- **Settings overview** — User account (username/password).
