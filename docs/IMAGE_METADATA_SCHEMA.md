# Image Metadata Schema Documentation

## Overview

Image metadata sidecars use a universal schema with optional type-specific fields. All images share core universal fields, while specific image types can include additional contextual information for more precise searchability.

## Schema Version

Current schema version: **1.0**

## Universal Fields (All Image Types)

These fields are available for ALL image types:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | Yes | Schema version (currently "1.0") |
| `image_filename` | string | Yes | Name of the image file |
| `type` | string | Yes | Image type (see types below) |
| `title` | string | Yes | Descriptive title for the image |
| `content` | string | Yes | Description/transcript of image contents |
| `author` | string | No | Author/creator name |
| `date` | string | No | Date in YYYY-MM-DD format |
| `series` | string | No | Series or collection name |
| `tags` | array | No | Array of searchable tags |

### Supported Image Types

- `comic` - Comic strips and panels
- `artwork` - Paintings, drawings, digital art
- `meme` - Internet memes
- `screenshot` - Software screenshots
- `medical` - Medical imaging (X-rays, MRI, CT, etc.)
- `documentation` - Technical documentation images
- `maps` - Geographic maps
- `photo` - Photographs
- `other` - Unclassified images

## Type-Specific Fields

### Photos (`type: "photo"`)

Additional fields for photographs:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `location` | string | Geographic location or venue | "Central Park, New York" |
| `event` | string | Event name | "Birthday Party 2024" |
| `coordinates` | string | GPS coordinates (lat,long) | "40.7128,-74.0060" |

**Example:**
```json
{
  "schema_version": "1.0",
  "image_filename": "family-reunion-2024.jpg",
  "type": "photo",
  "title": "Family Reunion 2024",
  "content": "Photo of the Smith family at the annual reunion gathering",
  "author": "Jane Smith",
  "date": "2024-07-15",
  "tags": ["family", "reunion", "summer"],
  "location": "Lake Tahoe, California",
  "event": "Smith Family Reunion 2024",
  "coordinates": "39.0968,-120.0324"
}
```

### Comics (`type: "comic"`)

Comics use universal fields with specific conventions:

| Field | Usage | Example |
|-------|-------|---------|
| `series` | Comic series name | "Dilbert" |
| `author` | Comic author/artist | "Scott Adams" |
| `content` | Panel-by-panel transcript | "Dilbert sits in his chair..." |

**Example:**
```json
{
  "schema_version": "1.0",
  "image_filename": "Dilbert - 1989-11-01.gif",
  "type": "comic",
  "title": "Dilbert - 1989-11-01",
  "content": "Dilbert sits in his chair watching television. The voice on the tv says, \"Tonight Siskel and Ebert review Dilbert's life.\" Ebert says, \". . . Boring and stupid . . . Look out, Gene; I'm gonna have to spit to get the taste out of my mouth . . .\" Ebert continues, \"Oops. Sorry, Gene.\" Dilbert points the remote control at the tv and changes the channel as he says, \"I hate when they do these theme shows.\"",
  "author": "Scott Adams",
  "date": "1989-11-01",
  "series": "Dilbert",
  "tags": ["dilbert", "tv", "cable tv", "remote", "boring", "stupid"]
}
```

### Artwork (`type: "artwork"`)

Additional fields for art pieces:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `medium` | string | Art medium/materials | "Oil on canvas" |
| `dimensions` | string | Physical dimensions | "24x36 inches" |

**Example:**
```json
{
  "schema_version": "1.0",
  "image_filename": "starry-night.jpg",
  "type": "artwork",
  "title": "The Starry Night",
  "content": "A swirling night sky with stars over a French village with a prominent cypress tree in the foreground",
  "author": "Vincent van Gogh",
  "date": "1889-06-01",
  "tags": ["post-impressionism", "landscape", "night scene", "van gogh"],
  "medium": "Oil on canvas",
  "dimensions": "29 x 36 1/4 inches"
}
```

### Medical Images (`type: "medical"`)

Additional fields for medical imaging:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `body_part` | string | Anatomical region | "chest", "skull", "knee" |
| `modality` | string | Imaging technology | "X-ray", "MRI", "CT", "Ultrasound" |

**Example:**
```json
{
  "schema_version": "1.0",
  "image_filename": "chest-xray-2024-01-15.jpg",
  "type": "medical",
  "title": "Chest X-Ray - Frontal View",
  "content": "PA chest radiograph showing clear lung fields with normal cardiac silhouette",
  "author": "Dr. Sarah Johnson",
  "date": "2024-01-15",
  "tags": ["radiology", "chest", "diagnostic", "x-ray"],
  "body_part": "chest",
  "modality": "X-ray"
}
```

### Maps (`type: "maps"`)

Additional fields for geographic maps:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `location` | string | Geographic area covered | "California State" |
| `map_type` | string | Map classification | "topographic", "political", "street" |
| `coordinates` | string | Center coordinates | "36.7783,-119.4179" |

**Example:**
```json
{
  "schema_version": "1.0",
  "image_filename": "yosemite-topo-map.jpg",
  "type": "maps",
  "title": "Yosemite National Park Topographic Map",
  "content": "USGS topographic map showing trails, elevations, and geographical features of Yosemite Valley",
  "author": "USGS",
  "date": "2023-05-01",
  "tags": ["topographic", "yosemite", "hiking", "trails", "elevation"],
  "location": "Yosemite National Park, California",
  "map_type": "topographic",
  "coordinates": "37.8651,-119.5383"
}
```

### Screenshots (`type: "screenshot"`)

Additional fields for software screenshots:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `application` | string | Software application | "VS Code", "Chrome", "Photoshop" |
| `platform` | string | Operating system | "Windows 11", "macOS Sonoma" |

**Example:**
```json
{
  "schema_version": "1.0",
  "image_filename": "vscode-python-debugging.png",
  "type": "screenshot",
  "title": "VS Code Python Debugging Session",
  "content": "Screenshot showing Python debugging in VS Code with breakpoint hit on line 42 of main.py",
  "date": "2024-01-28",
  "tags": ["vscode", "python", "debugging", "development"],
  "application": "Visual Studio Code",
  "platform": "Windows 11"
}
```

### Memes (`type: "meme"`)

Memes use universal fields:

**Example:**
```json
{
  "schema_version": "1.0",
  "image_filename": "distracted-boyfriend.jpg",
  "type": "meme",
  "title": "Distracted Boyfriend Meme - Python vs JavaScript",
  "content": "Meme showing boyfriend (labeled 'Developer') looking at another woman (labeled 'New Framework') while girlfriend (labeled 'Current Project') looks on disapprovingly",
  "date": "2024-01-20",
  "tags": ["meme", "programming", "humor", "development"]
}
```

### Documentation (`type: "documentation"`)

Documentation images use universal fields:

**Example:**
```json
{
  "schema_version": "1.0",
  "image_filename": "api-architecture-diagram.png",
  "type": "documentation",
  "title": "System Architecture Diagram - Microservices",
  "content": "Diagram showing the microservices architecture with gRPC communication between backend services, vector database, and frontend application",
  "author": "Engineering Team",
  "date": "2024-01-15",
  "series": "Technical Documentation v2.0",
  "tags": ["architecture", "microservices", "grpc", "system design"]
}
```

## Creating and Editing Metadata

### Via UI

1. Right-click on an image in the file tree
2. Select "Edit Image Metadata"
3. Fill in universal fields (required: type, title, content)
4. Add optional fields based on image type
5. Add searchable tags
6. Save

The system will create a `.metadata.json` sidecar file alongside your image.

### Via File Watcher

Place a `.metadata.json` file next to your image with the same base name:

```
myimage.jpg
myimage.jpg.metadata.json
```

The file watcher will automatically detect and process the metadata file.

### Manual Creation

Create a JSON file following the schema above and save it as `{image_filename}.metadata.json`.

## Search Benefits

All fields (universal + type-specific) are indexed for search:

- **Universal search**: Find images by title, content, author, series, or tags
- **Type-specific search**: 
  - Photos by location or event
  - Artwork by medium or dimensions
  - Medical images by body part or modality
  - Maps by map type or coordinates
  - Screenshots by application or platform

**Search Examples:**

- `"show me X-ray images of chest"` → Finds medical images with modality=X-ray and body_part=chest
- `"topographic maps of California"` → Finds maps with map_type=topographic and location containing California
- `"screenshots from VS Code on Windows"` → Finds screenshots with application=VS Code and platform=Windows
- `"artwork by Van Gogh in oil"` → Finds artwork with author=Van Gogh and medium containing oil

## Field Validation

- `date` must be in YYYY-MM-DD format
- `coordinates` should be in "latitude,longitude" format
- `tags` must be an array of strings
- All text fields support UTF-8 (international characters allowed)

## Notes

- Type-specific fields are optional but highly recommended for better searchability
- Empty/null fields are ignored during indexing
- File watcher automatically processes new `.metadata.json` files
- Updating a sidecar file triggers re-indexing
- Face detection can automatically add identity tags to photos
