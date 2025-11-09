(async function() {
  const container = document.getElementById('publications-container');
  const bibFiles = ['ASSETS.bib', 'ESANN.bib', 'EACL.bib', 'SDP.bib'];

  const parserURL = 'https://cdn.jsdelivr.net/npm/bibtex-parse-js@2.0.0/bibtexParse.min.js';
  const module = await import(parserURL);
  const bibtexParse = module.default || module;

  for (const file of bibFiles) {
    try {
      const res = await fetch(`./bibtex/${file}`);
      if (!res.ok) continue;
      const text = await res.text();
      const entries = bibtexParse.toJSON(text);
      if (!entries.length) continue;

      const e = entries[0];
      const t = e.entryTags || {};
      const id = (t.ID || file.split('.')[0]).toLowerCase();
      const title = t.title?.replace(/[{}]/g, '') || '';
      const authors = (t.author || '').replace(/ and /g, ', ');
      const year = t.year || '';
      const venue = t.booktitle || t.journal || '';
      const keywords = (t.keywords || '').split(',').map(k => k.trim()).filter(Boolean);

      const collapsed = `
        <section style="margin-top:0">
          <b>${authors}</b>.
          <span class="expandable highlight paper" id="${id}" style="margin-bottom:0">${title}</span>.
          ${venue ? venue + '.' : ''} ${year ? `(${year})` : ''}
        </section>
      `;

      const keywordHTML = keywords.length
        ? `<p>${keywords.map(k => `<span class="highlight-noexpand">${k}</span>`).join(' ')}</p>`
        : '';

      const expanded = `
        <section class="paper expansion" id="${id}">
          <div class="container">
            <span class="left-align">
              ${keywordHTML}
              ${t.abstract ? `<p><span class="highlight-noexpand">Abstract</span><br>${t.abstract}</p>` : ''}
            </span>
            <div style="flex-grow:1"></div>
            <span class="right-align" style="padding:0 5%;text-align:center;">
              ${t.url ? `<i>Click to view paper</i><br><a href="${t.url}" target="_blank"><img src="./images/${id.toUpperCase()}.png" style="height:200px;"></a><br>` : ''}
              <span class="highlight"><a href="./bibtex/${file}" download="${file}">Download BibTeX</a></span>
            </span>
          </div>
        </section>
      `;

      container.insertAdjacentHTML('beforeend', collapsed + expanded);
    } catch (err) {
      console.error(`Error loading ${file}:`, err);
    }
  }
})();
