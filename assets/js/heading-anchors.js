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

    var clearTimer = 0;

    anchor.addEventListener('click', function (event) {
      if (!navigator.clipboard || !navigator.clipboard.writeText) return; // plain link
      event.preventDefault();
      navigator.clipboard.writeText(absoluteUrl(heading.id)).then(
        function () {
          if (clearTimer) window.clearTimeout(clearTimer);
          anchor.setAttribute('data-copied', 'true');
          clearTimer = window.setTimeout(function () {
            anchor.removeAttribute('data-copied');
            clearTimer = 0;
          }, COPIED_MS);
        },
        function () {
          // The write was denied or otherwise failed. preventDefault() above
          // already stopped the browser's own navigation, so without this
          // the click would be a silent dead end (no feedback, and an
          // unhandled rejection). Fall back to an ordinary same-page
          // navigation instead, so the click still does something useful.
          window.location.hash = heading.id;
        }
      );
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
