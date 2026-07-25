(function () {
  'use strict';

  var COPIED_MS = 1200;

  function absoluteUrl(id) {
    return window.location.origin + window.location.pathname + '#' + id;
  }

  function decorate(heading, doc) {
    if (!heading.id || heading.querySelector('.heading-anchor')) return;

    var anchor = doc.createElement('a');
    anchor.className = 'heading-anchor';
    anchor.href = '#' + heading.id;
    anchor.textContent = '¶';
    anchor.setAttribute('aria-label', 'Copy link to this section');

    anchor.addEventListener('click', function (event) {
      if (!navigator.clipboard || !navigator.clipboard.writeText) return; // plain link
      event.preventDefault();
      navigator.clipboard.writeText(absoluteUrl(heading.id)).then(function () {
        anchor.setAttribute('data-copied', 'true');
        window.setTimeout(function () { anchor.removeAttribute('data-copied'); }, COPIED_MS);
      });
    });

    heading.appendChild(anchor);
  }

  function init(doc) {
    doc = doc || document;
    var content = doc.querySelector('.page-content');
    if (!content) return;
    var headings = content.querySelectorAll('h1[id], h2[id], h3[id]');
    for (var i = 0; i < headings.length; i++) decorate(headings[i], doc);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { init(document); });
  } else {
    init(document);
  }

  window.HeadingAnchors = { init: init };
})();
