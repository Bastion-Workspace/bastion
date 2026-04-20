---
title: Document encryption
order: 10
---

# Document encryption

You can **password-protect** individual text documents (`.md`, `.txt`, `.org`) so the file is stored as ciphertext on the server. This is useful for sensitive notes that should not be readable from disk backups or by anyone who does not know the password.

---

## What encryption does

- **At rest:** The file body is encrypted on the server. Bastion does not store your password in plain text; it stores verification material and a salt for key derivation.
- **Search:** Encrypted documents are **excluded from full-text search** while they remain encrypted. Removing encryption (with the correct password) restores normal indexing if your file is not otherwise exempt from search.
- **Processing:** Automatic embedding, entity extraction, and similar pipelines **do not run on ciphertext**. After you unlock in the editor and save, link extraction can run on the saved plaintext for the relation graph (for supported text types).
- **Collaboration:** Real-time collaborative editing is **not available** for encrypted files.
- **Exports / PDF / raw download:** Content endpoints require an **active unlock session** in the browser. If you see a “session required” or locked message, unlock the document first.

---

## Unlock session and multiple tabs

When you enter the correct password, the app opens an **unlock session** for that document only. The session is kept in memory in this browser tab (not in normal editor cache localStorage).

- **Switching document tabs:** If you leave an encrypted document tab open in the tab bar and switch to another document, your session can stay valid so you are not prompted again when you return—as long as the session has not expired and you have not locked the file.
- **Closing the last tab** for that document or clicking **Lock** ends the session on the server for that document.
- **Leaving the site** (closing the window or tab) attempts to end unlock sessions.
- **Each document has its own password and session.** Unlocking document A does not unlock document B. If one file references another in frontmatter or links, you still need a valid session (or password) for **each** encrypted target whose content you open.

---

## Where to encrypt or remove encryption

**From the folder tree**

- Right-click a `.md`, `.txt`, or `.org` file that is **not** encrypted: **Encrypt file…**
- Right-click an encrypted file: **Remove encryption…** (you will need the current password)

**From the editor**

- When a document is unlocked, the toolbar shows **Encrypted (unlocked)** and a **Lock** control to end the session immediately.

---

## Passwords and mistakes

- Choose a **strong password** you can recover (password manager recommended). If you lose the password, the content cannot be recovered from Bastion.
- **Encrypt** and **Remove encryption** flows ask for confirmation where applicable. Too many failed unlock attempts may be temporarily rate-limited for security.

---

## Logout and security notes

- **Logging out** clears in-memory unlock sessions tied to this integration.
- Encrypted content is only sent to your browser when you have unlocked with the correct password in this session. Do not share your password or leave an unlocked document unattended on a shared machine.

---

## Related help topics

Use the Help sidebar search for **Document Library overview**, **Markdown files**, and **Org Mode files** for editing and folder behavior.
