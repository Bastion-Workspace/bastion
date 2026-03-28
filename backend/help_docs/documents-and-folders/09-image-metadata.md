---
title: Image metadata and face tagging
order: 9
---

# Image metadata and face tagging

Images in the Document Library can have **searchable metadata** (type, title, author, description, etc.) and **face tags** that identify people in the image. You edit metadata from the **Edit Image Metadata** option on the image’s right-click menu; face tagging uses a **Face Tagger** to draw regions and assign names. Metadata and face tags are used by search and by agents. This page describes how to edit image metadata and use face tagging.

---

## Edit Image Metadata

- **Opening** — Right-click an **image file** in the Document Library sidebar and choose **Edit Image Metadata**. A dialog opens with tabs or sections for different kinds of metadata.
- **Fields** — You can set or change:
  - **Type** — Category of the image (e.g. Photo, Comic, Artwork, Meme, Screenshot, Medical, Documentation, Maps, Other). This helps search and filtering.
  - **Title** — A short title for the image.
  - **Author** — Creator or source.
  - **Date** — Date associated with the image.
  - **Description** — Longer description or caption. All of these fields are searchable so agents and search can find images by metadata.
- **Save** — Save your changes. The metadata is stored and indexed. You can close and reopen the dialog later to edit again.
- **LLM-assisted description** — If the instance supports it, a control may let you **generate** or **suggest** a description using an LLM (e.g. “Describe with LLM”). Use it to fill in description or other fields from the image content.

---

## Face tagging

- **What it is** — Face tagging associates **regions** in the image (faces) with **names** (e.g. person identities). Once tagged, you can search for images by person and agents can use that information.
- **Opening the Face Tagger** — From the image metadata dialog (or from the image viewer when available), use **Tag faces** or **Open Face Tagger**. The **Face Tagger** lets you draw one or more **regions** on the image and assign a name to each region. Names may be chosen from a list of known identities (e.g. from contacts or previous tags) or entered as new.
- **Face Tag Suggestions** — When the system detects **untagged faces** and has **suggestions** (e.g. from metadata or known identities), a **Face Tag Suggestions** panel may appear, offering to apply suggested names to untagged faces. You can accept or ignore suggestions and then refine in the Face Tagger.
- **Saving** — Save face tags from the Face Tagger or metadata dialog. Tags are stored with the document and used for search and by agents (e.g. “find images containing person X”).

---

## Object detection and annotation

- Some instances support **object detection** or **annotations** (e.g. drawing boxes or labels for objects in the image). If available, the image metadata dialog or a related component may offer an **Objects** or **Annotations** tab where you can add or edit regions and labels. Behavior depends on deployment; check the dialog for object/annotation options.

---

## How metadata and tags are used

- **Search** — Document search and vector search can use **metadata** (type, title, description, author) and **face tags** to find images. For example, you can search for “photos of Jane” or “comic artwork.”
- **Agents** — Agents that search or analyze documents can see image metadata and face tags and use them to answer questions or suggest edits (e.g. “add a caption” or “tag the person in this image”).

---

## Summary

- **Edit Image Metadata** (right-click on an image) lets you set **type**, **title**, **author**, **date**, and **description**. Optional **LLM-assisted** description may be available.
- **Face tagging** uses the **Face Tagger** to draw regions and assign names; **Face Tag Suggestions** can propose names for untagged faces. Tags are searchable and available to agents.
- **Object detection/annotation** may be available depending on instance. Metadata and face tags are used by **search** and **agents**.

---

## Related

- **Document Library overview** — Right-click options on files.
- **Editor features** — Viewing images and lightbox.
- **Folders** — Vectorization and search indexing.
- **Agent tools** — How agents access document metadata.
