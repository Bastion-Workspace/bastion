import React from 'react';
import {
  BarChart,
  AccountTree,
  Code,
  Image as ImageIcon,
  ViewInAr,
} from '@mui/icons-material';

/**
 * Icon for chat artifact types (ArtifactCard, collapsed rail, etc.).
 */
export function artifactTypeIcon(t) {
  switch ((t || '').toLowerCase()) {
    case 'chart':
      return <BarChart fontSize="small" color="primary" />;
    case 'mermaid':
      return <AccountTree fontSize="small" color="primary" />;
    case 'svg':
      return <ImageIcon fontSize="small" color="primary" />;
    case 'react':
      return <ViewInAr fontSize="small" color="primary" />;
    case 'html':
    default:
      return <Code fontSize="small" color="primary" />;
  }
}
