(async () => {
    // === CONFIGURATION ===
    // Concept record ID of Radiomics.jl on Zenodo ("Cite all versions"):
    // it is the trailing number of the concept DOI (10.5281/zenodo.<CONCEPT_RECID>).
    const CONCEPT_RECID = 'CONCEPT_RECID';
    // =====================

    const CONCEPT_DOI = `10.5281/zenodo.${CONCEPT_RECID}`;
    const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

    const $ = id => document.getElementById(id);

    // Escape text before injecting it into innerHTML
    const esc = s => String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');

    // "Last, First Second" -> "Last, F. S."
    const toApaName = name => {
        const [last, first = ''] = name.split(',').map(s => s.trim());
        const initials = first
            .split(/\s+/)
            .filter(Boolean)
            .map(p => p[0].toUpperCase() + '.')
            .join(' ');
        return initials ? `${last}, ${initials}` : last;
    };

    // Update the DOI text, links, badge image and badge Markdown snippet
    const setBadge = (doi, url) => {
        $('cite-doi').textContent = doi;
        $('cite-doi-link').href = url;
        $('cite-badge-link').href = url;
        $('cite-badge').src = `https://zenodo.org/badge/DOI/${doi}.svg`;
        $('cite-badge-md').textContent =
            `[![DOI](https://zenodo.org/badge/DOI/${doi}.svg)](${url})`;
    };

    try {
        // Ask Zenodo for the latest version of the record series
        const res = await fetch(
            `https://zenodo.org/api/records/${CONCEPT_RECID}/versions/latest`,
            { headers: { Accept: 'application/json' } }
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const rec = await res.json();
        const m = rec.metadata;

        const version = m.version;
        const doi = rec.doi || m.doi;
        const doiUrl = `https://doi.org/${doi}`;
        const title = m.title;

        const date = new Date(m.publication_date);
        const year = date.getUTCFullYear();
        const month = MONTHS[date.getUTCMonth()];

        // Authors (Zenodo returns them as "Last, First")
        const names = m.creators.map(c => c.name);
        const apaAuthors = names.map(toApaName);
        const authorStr = apaAuthors.length > 1
            ? apaAuthors.slice(0, -1).join(', ') + ', & ' + apaAuthors[apaAuthors.length - 1]
            : apaAuthors[0];

        // Version in the introductory paragraph
        $('cite-version').textContent = version;

        // Recommended citation (APA style)
        $('cite-apa').innerHTML =
            `${esc(authorStr)} (${year}). <em>${esc(title)}</em> ` +
            `(Version ${esc(version)}) [Computer software]. Zenodo. ` +
            `<a href="${esc(doiUrl)}">${esc(doiUrl)}</a>`;

        // BibTeX entry; the key is derived from the version string
        const key = 'radiomicsjl_' +
            version.replace(/[^a-zA-Z0-9]+/g, '_').replace(/^_+|_+$/g, '');

        $('cite-bibtex').textContent =
            `@software{${key},
  author    = {${names.join(' and ')}},
  title     = {${title}},
  month     = ${month},
  year      = ${year},
  publisher = {Zenodo},
  version   = {${version}},
  doi       = {${doi}},
  url       = {${doiUrl}}
}`;

        // DOI link and badge
        setBadge(doi, doiUrl);

    } catch (e) {
        // Zenodo unreachable: never show stale data, point to the concept DOI
        // (which always resolves to the latest release) instead
        console.warn('Zenodo unreachable, falling back to the concept DOI.', e);

        const url = `https://doi.org/${CONCEPT_DOI}`;

        $('cite-version').textContent = 'on Zenodo';
        $('cite-apa').innerHTML =
            `Unable to load the citation automatically. Please get the up-to-date ` +
            `citation for the latest release from <a href="${url}">${url}</a>.`;
        $('cite-bibtex').textContent =
            `% Unable to load the BibTeX entry automatically.\n` +
            `% Get it from ${url} (Export → BibTeX).`;
        setBadge(CONCEPT_DOI, url);
    }
})();