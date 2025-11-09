// generatePubs.js
(async function () {
  const container = document.getElementById('publications-container');
  const bibFiles = ['ASSETS.bib', 'ESANN.bib', 'EACL.bib', 'SDP.bib']; // expand later
  const parserURL = 'https://cdn.jsdelivr.net/npm/bibtex-parse-js@2.0.0/bibtexParse.min.js';

  // Load the BibTeX parser dynamically
  await import(parserURL).catch(err => console.error('BibTeX parser load error', err));

  for (const file of bibFiles) {
    try {
      const res = await fetch(`./bibtex/${file}`);
      const text = await res.text();
      const entry = bibtexParse.toJSON(text)[0];
      if (!entry) continue;

      const { entryTags: t } = entry;
      const id = (t.ID || file.split('.')[0]).toLowerCase();

      // Basic info
      const authors = (t.author || '').replace(/ and /g, ', ');
      const title = t.title?.replace(/[{}]/g, '');
      const year = t.year || '';
      const venue = t.booktitle || t.journal || '';

      // Keywords (optional)
      const keywords = (t.keywords || '').split(',').map(k => k.trim()).filter(Boolean);

      // Construct publication section (matching your expand/collapse structure)
      const shortHTML = `
        <section style="margin-top:0">
          <b>${authors}</b>.
          <span class="expandable highlight paper" id="${id}">${title}</span>.
          ${venue ? venue + '.' : ''} ${year ? `(${year})` : ''}
        </section>
      `;

      const keywordHTML = keywords.length
        ? `<p>${keywords.map(k => `<span class="highlight-noexpand">${k}</span>`).join(' ')}</p>`
        : '';

      const expandedHTML = `
        <section class="paper expansion" id="${id}">
          <div class="container">
            <span class="left-align">
              ${keywordHTML}
              <p><span class="highlight-noexpand">Abstract</span><br>${t.abstract || ''}</p>
            </span>
          </div>
        </section>
      `;

      container.insertAdjacentHTML('beforeend', shortHTML + expandedHTML);
    } catch (err) {
      console.error(`Error loading ${file}:`, err);
    }
  }
})();
