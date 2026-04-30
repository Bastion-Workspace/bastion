import type { OpdsFeedEntry, OpdsLink } from '../../api/ebooks';

export function linkLooksLikeOpdsCatalogNav(l: OpdsLink | undefined): boolean {
  if (!l?.href) return false;
  const rel = String(l.rel || '').toLowerCase();
  const typ = String(l.type || '').toLowerCase();
  if (rel.includes('opds-spec.org/acquisition') || (rel.includes('acquisition') && rel.includes('opds'))) {
    return false;
  }
  if (rel.includes('thumbnail') || rel.includes('cover') || rel.includes('opds-spec.org/image')) {
    return false;
  }
  if (rel.includes('subsection')) return true;
  if (rel.includes('opds-spec.org/catalog')) return true;
  if (rel.includes('opds-spec.org/group')) return true;
  if (typ.includes('profile=opds-catalog') || (typ.includes('opds-catalog') && typ.includes('atom'))) {
    return true;
  }
  if ((rel.includes('alternate') || rel.includes('related')) && typ.includes('opds-catalog')) {
    return true;
  }
  return false;
}

export function pickNavigationHrefFromEntry(en: OpdsFeedEntry): string {
  const fromApi = (en.navigation_links || []).find((l) => l?.href);
  if (fromApi?.href) return String(fromApi.href).trim();
  const fromLinks = (en.links || []).find(linkLooksLikeOpdsCatalogNav);
  return fromLinks?.href ? String(fromLinks.href).trim() : '';
}

export function pickAcquisitionHrefFromEntry(en: OpdsFeedEntry): string {
  if (en.acquisition_href) return String(en.acquisition_href).trim();
  const links = en.links || [];
  const acq = links.find((l) => {
    if (!l?.href) return false;
    const rel = String(l.rel || '').toLowerCase();
    const typ = String(l.type || '').toLowerCase();
    const isAcq =
      rel.includes('opds-spec.org/acquisition') || (rel.includes('acquisition') && rel.includes('opds'));
    if (!isAcq) return false;
    return typ.includes('epub') || String(l.href).toLowerCase().endsWith('.epub');
  });
  return acq?.href ? String(acq.href).trim() : '';
}

export function pickCoverHrefFromEntry(en: OpdsFeedEntry, baseUrl: string): string | null {
  const links = en.links || [];
  const cover = links.find((l) => {
    if (!l?.href) return false;
    const rel = String(l.rel || '').toLowerCase();
    return rel.includes('thumbnail') || rel.includes('cover') || rel.includes('opds-spec.org/image');
  });
  if (!cover?.href) return null;
  try {
    return new URL(String(cover.href).trim(), baseUrl).href;
  } catch {
    return null;
  }
}
