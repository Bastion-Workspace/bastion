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

export type AcquisitionPick = { href: string; format: 'epub' | 'pdf' };

function pickAcquisitionLinkFromEntry(en: OpdsFeedEntry): AcquisitionPick {
  if (en.acquisition_href) {
    const href = String(en.acquisition_href).trim();
    const t = String(en.acquisition_type || '').toLowerCase();
    const format: 'epub' | 'pdf' = t === 'pdf' ? 'pdf' : 'epub';
    return { href, format };
  }
  const links = en.links || [];
  const acq = links.find((l) => {
    if (!l?.href) return false;
    const rel = String(l.rel || '').toLowerCase();
    const typ = String(l.type || '').toLowerCase();
    const hrefL = String(l.href).toLowerCase();
    const isAcq =
      rel.includes('opds-spec.org/acquisition') || (rel.includes('acquisition') && rel.includes('opds'));
    if (!isAcq) return false;
    const isEpub = typ.includes('epub') || hrefL.endsWith('.epub');
    const isPdf = typ.includes('application/pdf') || hrefL.endsWith('.pdf');
    return isEpub || isPdf;
  });
  if (!acq?.href) return { href: '', format: 'epub' };
  const href = String(acq.href).trim();
  const typ = String(acq.type || '').toLowerCase();
  const hrefL = href.toLowerCase();
  const isEpub = typ.includes('epub') || hrefL.endsWith('.epub');
  const isPdf = typ.includes('application/pdf') || hrefL.endsWith('.pdf');
  const format: 'epub' | 'pdf' = isEpub ? 'epub' : isPdf ? 'pdf' : 'epub';
  return { href, format };
}

export function pickAcquisitionHrefFromEntry(en: OpdsFeedEntry): string {
  return pickAcquisitionLinkFromEntry(en).href;
}

export function pickAcquisitionFormatFromEntry(en: OpdsFeedEntry): 'epub' | 'pdf' {
  return pickAcquisitionLinkFromEntry(en).format;
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
