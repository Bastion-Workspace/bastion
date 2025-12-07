/**
 * Utility functions for detecting and extracting embeddable content from URLs
 */

/**
 * Extract YouTube video ID from various YouTube URL formats
 * Supports:
 * - https://www.youtube.com/watch?v=VIDEO_ID
 * - https://youtu.be/VIDEO_ID
 * - https://www.youtube.com/embed/VIDEO_ID
 * - https://m.youtube.com/watch?v=VIDEO_ID
 */
export function extractYouTubeVideoId(url) {
  if (!url || typeof url !== 'string') return null;
  
  // Remove any trailing slashes or query params after video ID
  const cleanUrl = url.trim();
  
  // Pattern 1: youtube.com/watch?v=VIDEO_ID or youtube.com/watch?feature=...&v=VIDEO_ID
  const watchMatch = cleanUrl.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/);
  if (watchMatch) {
    return watchMatch[1];
  }
  
  // Pattern 2: youtube.com/v/VIDEO_ID
  const vMatch = cleanUrl.match(/youtube\.com\/v\/([a-zA-Z0-9_-]{11})/);
  if (vMatch) {
    return vMatch[1];
  }
  
  return null;
}

/**
 * Check if a URL is a YouTube URL
 */
export function isYouTubeUrl(url) {
  if (!url || typeof url !== 'string') return false;
  return /(?:youtube\.com|youtu\.be)/.test(url);
}

/**
 * Extract Rumble video ID from various Rumble URL formats
 * Supports:
 * - https://rumble.com/vVIDEO_ID-title.html
 * - https://rumble.com/embed/vVIDEO_ID/
 * - https://rumble.com/vVIDEO_ID/
 */
export function extractRumbleVideoId(url) {
  if (!url || typeof url !== 'string') return null;
  
  const cleanUrl = url.trim();
  
  // Pattern 1: rumble.com/vVIDEO_ID or rumble.com/vVIDEO_ID-title.html
  // Rumble video IDs start with 'v' followed by alphanumeric characters
  const vMatch = cleanUrl.match(/rumble\.com\/(?:embed\/)?v([a-zA-Z0-9]+)/);
  if (vMatch) {
    return vMatch[1];
  }
  
  return null;
}

/**
 * Check if a URL is a Rumble URL
 */
export function isRumbleUrl(url) {
  if (!url || typeof url !== 'string') return false;
  return /rumble\.com/.test(url);
}

/**
 * Parse text content and extract URLs that can be embedded
 * Returns an array of segments: { type: 'text'|'embed', content: string, embedData?: object }
 */
export function parseContentForEmbeds(text) {
  if (!text || typeof text !== 'string') {
    return [{ type: 'text', content: text || '' }];
  }
  
  // URL regex pattern - matches http/https URLs
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const segments = [];
  let lastIndex = 0;
  let match;
  
  while ((match = urlRegex.exec(text)) !== null) {
    // Add text before the URL
    if (match.index > lastIndex) {
      segments.push({
        type: 'text',
        content: text.substring(lastIndex, match.index)
      });
    }
    
    const url = match[0];
    const youtubeId = extractYouTubeVideoId(url);
    const rumbleId = extractRumbleVideoId(url);
    
    if (youtubeId) {
      // YouTube embed
      segments.push({
        type: 'embed',
        embedType: 'youtube',
        content: url,
        embedData: {
          videoId: youtubeId,
          url: url
        }
      });
    } else if (rumbleId) {
      // Rumble embed
      segments.push({
        type: 'embed',
        embedType: 'rumble',
        content: url,
        embedData: {
          videoId: rumbleId,
          url: url
        }
      });
    } else {
      // Regular URL - keep as text (will be rendered as link)
      segments.push({
        type: 'text',
        content: url
      });
    }
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < text.length) {
    segments.push({
      type: 'text',
      content: text.substring(lastIndex)
    });
  }
  
  // If no URLs found, return entire text as single segment
  if (segments.length === 0) {
    return [{ type: 'text', content: text }];
  }
  
  return segments;
}

