/**
 * Help Topics Index
 * 
 * To add a new help topic:
 * 1. Create a new markdown file in frontend/src/help/ with frontmatter:
 *    ---
 *    title: Your Topic Title
 *    ---
 *    
 *    Your markdown content here...
 * 
 * 2. Add an entry to the helpTopics array below with:
 *    - id: filename without .md extension
 *    - title: from frontmatter (or custom)
 *    - content: markdown content (copy from .md file, remove frontmatter)
 * 
 * The .md files serve as your source of truth for editing.
 * Copy the content (without frontmatter) into the content field here.
 */

import { APP_VERSION } from '../config/version';

export const helpTopics = [
  {
    id: 'about',
    title: 'About',
    content: `# About Bastion

**Version**: ${APP_VERSION}

Bastion is a comprehensive personal workspace platform featuring specialized agents for knowledge management, creative writing, data engineering, collaboration, and productivity.

## Project Repository

[View on GitHub](https://github.com/adamsih300u/bastion)

## License

Apache-2.0 License - See LICENSE file for details.`,
  },
  // Add more help topics here as needed
  // Example:
  // {
  //   id: 'getting-started',
  //   title: 'Getting Started',
  //   content: `# Getting Started
  // 
  // Welcome to Bastion! This guide will help you get started...
  // 
  // ## First Steps
  // 
  // 1. Upload documents
  // 2. Create folders
  // 3. Start chatting with agents
  // `,
  // },
];
