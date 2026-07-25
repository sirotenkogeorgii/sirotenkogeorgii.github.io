(function () {
  'use strict';

  var MIN_ENTRIES = 4;

  function stripMath(text) {
    // Unwrap $...$ / $$...$$ delimiters and drop TeX punctuation ({ } ^ _ and the
    // backslash itself), but keep the command name (e.g. "\pi" -> "pi", "\ell" ->
    // "ell") so a heading rendered as $V^\pi$ is still findable by typing "pi".
    return String(text)
      .replace(/\$\$?/g, ' ')
      .replace(/[\\{}^_]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
  }

  function levelOf(li, root) {
    var level = 1;
    var node = li.parentElement;
    while (node && node !== root) {
      if (node.tagName === 'UL') level++;
      node = node.parentElement;
    }
    return level;
  }

  function collectEntries(tocRoot, doc) {
    var entries = [];
    if (!tocRoot) return entries;
    var anchors = tocRoot.querySelectorAll('li > a[href^="#"]');
    for (var i = 0; i < anchors.length; i++) {
      var anchor = anchors[i];
      var id;
      try {
        id = decodeURIComponent(anchor.getAttribute('href').slice(1));
      } catch (e) {
        id = anchor.getAttribute('href').slice(1);
      }
      var heading = id ? doc.getElementById(id) : null;
      if (!heading) continue;
      var text = anchor.textContent.trim();
      entries.push({
        id: id,
        level: levelOf(anchor.parentElement, tocRoot),
        text: text,
        filterKey: stripMath(text),
        anchorEl: anchor,
        liEl: anchor.parentElement,
        headingEl: heading
      });
    }
    return entries;
  }

  function buildTree(entries) {
    var roots = [];
    var stack = [];
    entries.forEach(function (entry) {
      var node = { entry: entry, children: [] };
      while (stack.length && stack[stack.length - 1].entry.level >= entry.level) stack.pop();
      if (stack.length) stack[stack.length - 1].children.push(node);
      else roots.push(node);
      stack.push(node);
    });
    return roots;
  }

  function shouldActivate(entries) {
    return !!entries && entries.length >= MIN_ENTRIES;
  }

  function decorate(entries, tocRoot) {
    tocRoot.classList.add('toc-panel__list');
    entries.forEach(function (entry) {
      var li = entry.liEl;
      li.classList.add('toc-panel__item', 'toc-panel__item--l' + entry.level);
      li.setAttribute('data-toc-id', entry.id);
      entry.anchorEl.classList.add('toc-panel__link');

      var sublist = li.querySelector(':scope > ul');
      if (!sublist) return;

      sublist.classList.add('toc-panel__sublist');
      li.classList.add('toc-panel__item--branch');
      li.setAttribute('data-expanded', 'false');

      var twisty = document.createElement('button');
      twisty.type = 'button';
      twisty.className = 'toc-panel__twisty';
      twisty.setAttribute('aria-expanded', 'false');
      twisty.setAttribute('aria-label', 'Toggle ' + entry.text);
      li.insertBefore(twisty, entry.anchorEl);
    });
  }

  function hideLegacyCaption(tocRoot) {
    var prev = tocRoot.previousElementSibling;
    if (!prev || prev.tagName !== 'P') return;
    var strong = prev.querySelector('strong');
    if (strong && strong.textContent.trim() === 'Table of Contents') {
      prev.classList.add('toc-legacy-caption');
    }
  }

  function renderShell(doc) {
    var root = doc.createElement('div');
    root.className = 'toc-root';

    var hotzone = doc.createElement('div');
    hotzone.className = 'toc-root__hotzone';

    var rail = doc.createElement('div');
    rail.className = 'toc-rail';
    rail.setAttribute('aria-hidden', 'true');

    var panel = doc.createElement('nav');
    panel.className = 'toc-panel';
    panel.setAttribute('role', 'navigation');
    panel.setAttribute('aria-label', 'Table of contents');

    var header = doc.createElement('div');
    header.className = 'toc-panel__header';

    var filter = doc.createElement('input');
    filter.type = 'search';
    filter.className = 'toc-panel__filter';
    filter.placeholder = 'Filter sections…';
    filter.setAttribute('aria-label', 'Filter sections');
    filter.autocomplete = 'off';

    var body = doc.createElement('div');
    body.className = 'toc-panel__body';

    header.appendChild(filter);
    panel.appendChild(header);
    panel.appendChild(body);
    root.appendChild(hotzone);
    root.appendChild(rail);
    root.appendChild(panel);

    return { root: root, hotzone: hotzone, rail: rail, panel: panel, filterEl: filter, bodyEl: body };
  }

  function createScrollSpy(entries, contentEl, onChange) {
    var offsets = [];
    var activeId = null;
    var frame = 0;

    function measure() {
      offsets = entries.map(function (entry) {
        return {
          id: entry.id,
          top: entry.headingEl.getBoundingClientRect().top + window.pageYOffset
        };
      });
      offsets.sort(function (a, b) { return a.top - b.top; });
    }

    function pick() {
      frame = 0;
      if (!offsets.length) return;

      var probe = window.pageYOffset + window.innerHeight * 0.2;
      var found = offsets[0].id;
      for (var i = 0; i < offsets.length; i++) {
        if (offsets[i].top <= probe) found = offsets[i].id;
        else break;
      }

      var atBottom =
        window.innerHeight + window.pageYOffset >=
        document.documentElement.scrollHeight - 2;
      if (atBottom) found = offsets[offsets.length - 1].id;

      if (found !== activeId) {
        activeId = found;
        onChange(found);
      }
    }

    function schedule() {
      if (!frame) frame = window.requestAnimationFrame(pick);
    }

    measure();
    pick();

    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', function () { measure(); schedule(); });

    // MathJax reflows the page long after load; watch the content box rather
    // than depending on MathJax's own promises.
    if (window.ResizeObserver && contentEl) {
      new window.ResizeObserver(function () { measure(); schedule(); }).observe(contentEl);
    }

    return {
      measure: measure,
      refresh: function () { measure(); pick(); },
      get activeId() { return activeId; }
    };
  }

  function renderTicks(railEl, entries, doc) {
    var ticks = {};
    entries.forEach(function (entry) {
      if (entry.level !== 1) return;
      var tick = doc.createElement('span');
      tick.className = 'toc-rail__tick';
      tick.setAttribute('data-toc-id', entry.id);
      railEl.appendChild(tick);
      ticks[entry.id] = tick;
    });

    function position() {
      var height = Math.max(document.documentElement.scrollHeight, 1);
      Object.keys(ticks).forEach(function (id) {
        var heading = document.getElementById(id);
        if (!heading) return;
        var top = heading.getBoundingClientRect().top + window.pageYOffset;
        var pct = Math.min(100, Math.max(0, (top / height) * 100));
        ticks[id].style.top = pct.toFixed(3) + '%';
      });
    }

    position();
    return { ticks: ticks, position: position };
  }

  function topLevelAncestorId(listEl, id) {
    var li = listEl.querySelector('li[data-toc-id="' + cssEscape(id) + '"]');
    var top = null;
    while (li && li !== listEl) {
      if (li.tagName === 'LI' && li.hasAttribute('data-toc-id')) top = li;
      li = li.parentElement;
    }
    return top ? top.getAttribute('data-toc-id') : null;
  }

  function cssEscape(value) {
    if (window.CSS && window.CSS.escape) return window.CSS.escape(value);
    return String(value).replace(/["\\]/g, '\\$&');
  }

  function init(doc) {
    doc = doc || document;
    var tocRoot = doc.getElementById('markdown-toc');
    if (!tocRoot) return null;

    var entries = collectEntries(tocRoot, doc);
    if (!shouldActivate(entries)) return null;

    decorate(entries, tocRoot);
    hideLegacyCaption(tocRoot);

    var shell = renderShell(doc);
    shell.bodyEl.appendChild(tocRoot);   // move, do not clone: MathJax typesets once
    doc.body.appendChild(shell.root);
    doc.documentElement.setAttribute('data-toc', 'on');

    var instance = {
      entries: entries,
      tree: buildTree(entries),
      root: shell.root,
      hotzone: shell.hotzone,
      rail: shell.rail,
      panel: shell.panel,
      listEl: tocRoot,
      filterEl: shell.filterEl
    };
    window.TocPanel.instance = instance;

    var rails = renderTicks(shell.rail, entries, doc);
    instance.ticks = rails;

    instance.spy = createScrollSpy(
      entries,
      doc.querySelector('.page-content') || doc.body,
      function (id) {
        var previous = instance.listEl.querySelector('li[data-active="true"]');
        if (previous) previous.removeAttribute('data-active');

        var current = instance.listEl.querySelector('li[data-toc-id="' + cssEscape(id) + '"]');
        if (current) current.setAttribute('data-active', 'true');

        var topId = topLevelAncestorId(instance.listEl, id);
        Object.keys(rails.ticks).forEach(function (tickId) {
          rails.ticks[tickId].classList.toggle('toc-rail__tick--active', tickId === topId);
        });

        rails.position();
        doc.dispatchEvent(new CustomEvent('toc:active', { detail: { id: id, topId: topId } }));
      }
    );

    return instance;
  }

  window.TocPanel = {
    MIN_ENTRIES: MIN_ENTRIES,
    stripMath: stripMath,
    collectEntries: collectEntries,
    buildTree: buildTree,
    shouldActivate: shouldActivate,
    init: init,
    instance: null
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { init(document); });
  } else {
    init(document);
  }
})();
