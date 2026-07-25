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

    var backdrop = doc.createElement('div');
    backdrop.className = 'toc-backdrop';

    var fab = doc.createElement('button');
    fab.type = 'button';
    fab.className = 'toc-fab';
    fab.setAttribute('aria-expanded', 'false');
    fab.setAttribute('aria-label', 'Open table of contents');
    fab.innerHTML =
      '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" ' +
      'stroke-width="2" stroke-linecap="round" aria-hidden="true" focusable="false">' +
      '<line x1="4" y1="7" x2="20" y2="7"></line>' +
      '<line x1="4" y1="12" x2="20" y2="12"></line>' +
      '<line x1="4" y1="17" x2="14" y2="17"></line></svg>';

    root.insertBefore(backdrop, panel);
    root.appendChild(fab);

    return {
      root: root,
      hotzone: hotzone,
      rail: rail,
      panel: panel,
      filterEl: filter,
      bodyEl: body,
      backdrop: backdrop,
      fab: fab
    };
  }

  function createScrollSpy(entries, contentEl, onChange, onMeasure) {
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
      // Tick positions depend only on document height and heading offsets,
      // both invariant under pure scrolling, so recompute them wherever
      // measure() runs (initial load, resize, MathJax reflow) instead of on
      // every scroll-driven active-heading change.
      if (onMeasure) onMeasure();
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

      // Only defer to "last heading wins" when the page actually has room to
      // scroll; otherwise a short page (fits on one screen) reports atBottom
      // true at pageYOffset 0 and the last heading would be wrongly active
      // before the reader has scrolled at all.
      var scrollRange = document.documentElement.scrollHeight - window.innerHeight;
      var atBottom = scrollRange > 2 && window.pageYOffset >= scrollRange - 2;
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
      var ids = Object.keys(ticks);
      // Batch every getBoundingClientRect() read before any style.top write
      // so this never interleaves reads and writes per tick (layout thrash).
      var pcts = ids.map(function (id) {
        var heading = document.getElementById(id);
        if (!heading) return null;
        var top = heading.getBoundingClientRect().top + window.pageYOffset;
        return Math.min(100, Math.max(0, (top / height) * 100));
      });
      ids.forEach(function (id, i) {
        if (pcts[i] === null) return;
        ticks[id].style.top = pcts[i].toFixed(3) + '%';
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

  function createAccordion(listEl) {
    var lastTopId = null;

    function branches() {
      return listEl.querySelectorAll('li.toc-panel__item--branch');
    }

    function setExpanded(li, expanded) {
      li.setAttribute('data-expanded', expanded ? 'true' : 'false');
      var twisty = li.querySelector(':scope > .toc-panel__twisty');
      if (twisty) twisty.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    }

    function ancestorChain(id) {
      var chain = [];
      var node = listEl.querySelector('li[data-toc-id="' + cssEscape(id) + '"]');
      while (node && node !== listEl) {
        if (node.tagName === 'LI' && node.classList.contains('toc-panel__item--branch')) {
          chain.push(node);
        }
        node = node.parentElement;
      }
      return chain;
    }

    function clearLocks() {
      var locked = listEl.querySelectorAll('li[data-user-locked="true"]');
      for (var i = 0; i < locked.length; i++) locked[i].removeAttribute('data-user-locked');
    }

    function syncTo(id, topId) {
      if (topId !== lastTopId) {
        clearLocks();
        lastTopId = topId;
      }

      var chain = ancestorChain(id);
      var all = branches();
      for (var i = 0; i < all.length; i++) {
        var li = all[i];
        if (li.getAttribute('data-user-locked') === 'true') continue;
        setExpanded(li, chain.indexOf(li) !== -1);
      }
    }

    listEl.addEventListener('click', function (event) {
      var twisty = event.target.closest ? event.target.closest('.toc-panel__twisty') : null;
      if (!twisty) return;
      event.preventDefault();
      var li = twisty.parentElement;
      var next = li.getAttribute('data-expanded') !== 'true';
      setExpanded(li, next);
      li.setAttribute('data-user-locked', 'true');
    });

    function expandAll() {
      var all = branches();
      for (var i = 0; i < all.length; i++) setExpanded(all[i], true);
    }

    function restore() {
      lastTopId = null;
      clearLocks();
    }

    return { syncTo: syncTo, expandAll: expandAll, restore: restore, setExpanded: setExpanded };
  }

  var CLOSE_DELAY = 250;
  var MOBILE_QUERY = '(max-width: 899px)';

  function createReveal(instance, doc) {
    var rootEl = doc.documentElement;
    var mobileMedia = window.matchMedia(MOBILE_QUERY);
    var timer = 0;

    // Two modes only. The panel is never pinned open at any width: it stays
    // shut until the pointer reaches the right edge and slides back out of the
    // way when the pointer leaves. That is the whole point of the feature —
    // the reading column is never permanently narrowed by a sidebar.
    function mode() {
      return mobileMedia.matches ? 'mobile' : 'overlay';
    }

    function setOpen(next) {
      rootEl.setAttribute('data-toc-open', next ? 'true' : 'false');
      instance.fab.setAttribute('aria-expanded', next ? 'true' : 'false');
      instance.fab.setAttribute(
        'aria-label',
        next ? 'Close table of contents' : 'Open table of contents'
      );
      // createKeyboard (instance.keyboard) is wired up after createReveal —
      // see the ordering comment above instance.reveal's assignment in init()
      // — and this very branch already runs once during that earlier
      // construction (applyMode()'s initial overlay-mode close), before
      // instance.keyboard exists. A doc-level event, the same pattern as
      // 'toc:active' below, lets the two modules stay decoupled either way:
      // this fires on every close regardless of whether a listener exists
      // yet, and createKeyboard attaches clearHighlight() to it whenever it
      // is constructed, in either order, with no reference back into
      // instance.reveal required.
      if (!next) doc.dispatchEvent(new CustomEvent('toc:closed'));
    }

    function applyMode() {
      var next = mode();
      rootEl.setAttribute('data-toc-mode', next);
      if (next === 'mobile') {
        // Crossing into mobile with data-toc-open still "true" means an open
        // drawer *plus* the modal backdrop dimming and swallowing taps on the
        // article, and there is no mouse-leave to undo that on a touch device.
        close(true);
      } else if (rootEl.getAttribute('data-toc-open') !== 'true') {
        setOpen(false);
      }
    }

    function cancelClose() {
      if (timer) { window.clearTimeout(timer); timer = 0; }
    }

    function open() {
      cancelClose();
      setOpen(true);
    }

    function close(immediate) {
      cancelClose();
      if (immediate) {
        setOpen(false);
      } else {
        timer = window.setTimeout(function () {
          timer = 0;
          setOpen(false);
        }, CLOSE_DELAY);
      }
    }

    function toggle() {
      if (rootEl.getAttribute('data-toc-open') === 'true') close(true);
      else open();
    }

    instance.hotzone.addEventListener('mouseenter', function () {
      if (mode() === 'mobile') return;
      open();
    });
    instance.rail.addEventListener('mouseenter', function () {
      if (mode() === 'mobile') return;
      open();
    });
    instance.rail.addEventListener('mouseleave', function () { close(); });
    instance.panel.addEventListener('mouseenter', cancelClose);
    instance.panel.addEventListener('mouseleave', function () { close(); });

    instance.listEl.addEventListener('click', function (event) {
      var link = event.target.closest ? event.target.closest('.toc-panel__link') : null;
      if (!link) return;
      close(true);
    });

    instance.fab.addEventListener('click', function () { toggle(); });
    instance.backdrop.addEventListener('click', function () { close(true); });

    applyMode();
    if (mobileMedia.addEventListener) {
      mobileMedia.addEventListener('change', applyMode);
    } else if (mobileMedia.addListener) {
      mobileMedia.addListener(applyMode);
    }

    return { open: open, close: close, toggle: toggle, mode: mode };
  }

  function createFilter(instance, doc) {
    var listEl = instance.listEl;
    var rootEl = doc.documentElement;
    var active = false;

    function markOut(li, out) {
      if (out) li.setAttribute('data-filtered', 'out');
      else li.removeAttribute('data-filtered');
    }

    function clear() {
      var items = listEl.querySelectorAll('li[data-toc-id]');
      for (var i = 0; i < items.length; i++) markOut(items[i], false);
      rootEl.setAttribute('data-toc-filtering', 'false');
      active = false;
      instance.accordion.restore();
      // accordion.restore() only clears user-locks and lastTopId; it does not
      // collapse anything, so on its own it would leave every branch expanded
      // from the apply()-time expandAll() above. spy.refresh() cannot fix
      // that either: pick() only invokes onChange (which calls
      // accordion.syncTo) when the active heading id changes, and clearing a
      // filter does not move the reader, so the id is unchanged and
      // onChange never fires. Re-sync explicitly against the reader's actual
      // position instead.
      if (instance.spy) {
        var activeId = instance.spy.activeId;
        if (activeId) instance.accordion.syncTo(activeId, topLevelAncestorId(listEl, activeId));
      }
    }

    function apply(query) {
      var needle = stripMath(query);
      if (!needle) { clear(); return; }

      active = true;
      rootEl.setAttribute('data-toc-filtering', 'true');

      var keep = {};
      instance.entries.forEach(function (entry) {
        if (entry.filterKey.indexOf(needle) === -1) return;
        keep[entry.id] = true;
        // Keep every ancestor so the match has context.
        var node = entry.liEl.parentElement;
        while (node && node !== listEl) {
          if (node.tagName === 'LI' && node.hasAttribute('data-toc-id')) {
            keep[node.getAttribute('data-toc-id')] = true;
          }
          node = node.parentElement;
        }
      });

      var items = listEl.querySelectorAll('li[data-toc-id]');
      for (var i = 0; i < items.length; i++) {
        markOut(items[i], !keep[items[i].getAttribute('data-toc-id')]);
      }
      instance.accordion.expandAll();
    }

    function matches() {
      return Array.prototype.filter.call(
        listEl.querySelectorAll('a.toc-panel__link'),
        function (a) { return a.getBoundingClientRect().height > 0; }
      );
    }

    instance.filterEl.addEventListener('input', function () {
      apply(instance.filterEl.value);
    });

    // Public hook, not dead code: no rule in site.css keys off
    // data-toc-filtering; it is the page-level "a filter is running" signal
    // for user styles and tests, and is asserted by tests/toc-filter.spec.mjs.
    rootEl.setAttribute('data-toc-filtering', 'false');

    return {
      apply: apply,
      clear: function () { instance.filterEl.value = ''; clear(); },
      matches: matches,
      get active() { return active; }
    };
  }

  // Types that are never genuine text entry. A focused <input type="range">
  // or type="checkbox"> cannot receive typed characters at all, so treating
  // it as "typing" only blocks the shortcuts below with nothing to show for
  // it — this site's note pages carry plenty of both (e.g. subpages/books/
  // sde_hd and subpages/books/pdeds, each with a {:toc} and a dozen-plus
  // range sliders). A missing or unrecognised type falls through to the
  // default below, matching how the browser itself treats a bare <input>.
  var NON_TEXT_INPUT_TYPES = {
    range: true,
    checkbox: true,
    radio: true,
    button: true,
    submit: true,
    reset: true,
    color: true,
    file: true
  };

  function isTypingTarget(el) {
    if (!el) return false;
    var tag = el.tagName;
    if (tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable === true) return true;
    return tag === 'INPUT' && !NON_TEXT_INPUT_TYPES[el.type];
  }

  function createKeyboard(instance, doc) {
    var cursor = -1;
    var rowsCache = null;

    // instance.filter.matches() calls getBoundingClientRect() on every
    // a.toc-panel__link to test visibility (Task 8) — up to ~700 rows on the
    // largest real page. rows() is on the hot path of every arrow keypress,
    // so calling matches() straight through here would turn cursor-walking
    // into an O(rows) layout read per key. Cache the row list instead.
    //
    // The visible set changes exactly when one of two attributes flips on a
    // row's ancestry: data-filtered (the filter, Task 8) or data-expanded
    // (the accordion, Task 5) — see the matching CSS rules at the bottom of
    // site.css. data-expanded does not only change on this module's own
    // open/close: createScrollSpy's onChange calls accordion.syncTo() on
    // every scroll-driven active-heading change (the ordinary way to use a
    // pinned panel while reading), and createAccordion's own click listener
    // toggles a twisty independently of both reveal and keyboard. Rather
    // than hand-enumerate every call site that can flip those two
    // attributes (and silently go stale the next time createAccordion
    // changes), watch instance.listEl directly and invalidate on any
    // matching mutation, wherever it comes from.
    function invalidateRows() { rowsCache = null; }

    function rows() {
      if (!rowsCache) rowsCache = instance.filter.matches();
      return rowsCache;
    }

    if (window.MutationObserver) {
      new window.MutationObserver(invalidateRows).observe(instance.listEl, {
        attributes: true,
        subtree: true,
        attributeFilter: ['data-expanded', 'data-filtered']
      });
    }

    // The outline (data-kbd) and the cursor it indexes must always be dropped
    // together. Leaving either behind across a close means the next session
    // starts with a stale selection: a row is visibly outlined while Enter
    // navigates to whatever now sits at the old cursor index in a row list
    // that has since been rebuilt (filter cleared, branches re-collapsed).
    function clearHighlight() {
      var previous = instance.listEl.querySelector('li[data-kbd="true"]');
      if (previous) previous.removeAttribute('data-kbd');
      cursor = -1;
    }

    // Every close path that is not Escape — mouseleave, a backdrop tap, or
    // clicking a TOC link — goes through createReveal's setOpen(false), never
    // through this module directly, so it cannot call clearHighlight() here
    // without instance.keyboard reaching back into instance.reveal (or vice
    // versa). Listening for the event setOpen(false) dispatches instead keeps
    // the two modules independent of each other's construction order; see the
    // comment at that dispatch site for why an event and not a direct call.
    doc.addEventListener('toc:closed', clearHighlight);

    function highlight(index) {
      var list = rows();
      clearHighlight();

      if (!list.length) return;

      cursor = Math.max(0, Math.min(index, list.length - 1));
      var li = list[cursor].parentElement;
      li.setAttribute('data-kbd', 'true');
      if (li.scrollIntoView) li.scrollIntoView({ block: 'nearest' });
    }

    function openForSearch() {
      invalidateRows();
      clearHighlight();
      instance.reveal.open();
      instance.filterEl.focus();
      instance.filterEl.select();
    }

    doc.addEventListener('keydown', function (event) {
      var typing = isTypingTarget(doc.activeElement);
      var inFilter = doc.activeElement === instance.filterEl;

      if (!typing && event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault();
        openForSearch();
        return;
      }

      // Same !typing guard as the '/' branch above, with one exception:
      // focus already inside the panel's own filter (inFilter) must not be
      // blocked by it, or Cmd/Ctrl+K becomes a no-op while the reader is
      // filtering instead of re-running openForSearch (re-focus, re-select).
      // Cmd/Ctrl+K must still not be stolen from an input the reader is
      // genuinely typing in otherwise; that it has never misfired there is an
      // accident of deployment (the only page with a search box has no
      // #markdown-toc, so this listener is never bound there), not a
      // property of this code.
      if ((!typing || inFilter) && (event.metaKey || event.ctrlKey) && (event.key === 'k' || event.key === 'K')) {
        event.preventDefault();
        openForSearch();
        return;
      }

      if (event.key === 'Escape') {
        if (instance.filter.active) instance.filter.clear();
        instance.reveal.close(true);
        invalidateRows();
        clearHighlight();
        if (inFilter) instance.filterEl.blur();
        return;
      }

      if (!inFilter) return;

      if (event.key === 'ArrowDown') {
        event.preventDefault();
        highlight(cursor + 1);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        highlight(cursor - 1);
      } else if (event.key === 'Enter') {
        event.preventDefault();
        var list = rows();
        var target = list[cursor >= 0 ? cursor : 0];
        if (target) target.click();
      }
    });

    instance.filterEl.addEventListener('input', function () { invalidateRows(); highlight(0); });

    return { highlight: highlight };
  }

  function init(doc) {
    // Idempotent: a second call would build a second .toc-root, a second set
    // of twisties inside the same <ul>, and a second scroll-spy/ResizeObserver
    // pair fighting the first over data-active. No production path calls init
    // twice, but nothing stopped one from doing so either.
    if (window.TocPanel.instance) return window.TocPanel.instance;

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
    // Public hook, not dead code: data-toc="on" is the page-level signal that
    // the panel took over, for user styles/bookmarklets and for debugging a
    // live page. No rule in site.css keys off it by design.
    doc.documentElement.setAttribute('data-toc', 'on');

    var instance = {
      entries: entries,
      // Public hook, not dead code: nothing inside this module reads the tree
      // (rendering reuses kramdown's own nesting), but buildTree is exported
      // and pinned by tests as the model layer's shape contract.
      tree: buildTree(entries),
      root: shell.root,
      hotzone: shell.hotzone,
      rail: shell.rail,
      panel: shell.panel,
      listEl: tocRoot,
      filterEl: shell.filterEl,
      backdrop: shell.backdrop,
      fab: shell.fab
    };
    window.TocPanel.instance = instance;
    // Kept ahead of renderTicks/createAccordion/createScrollSpy below, which
    // call getBoundingClientRect() and so force a synchronous layout pass that
    // commits .toc-panel's off-screen translateX(100%) as an already-rendered
    // style. createReveal()'s applyMode() writes data-toc-mode/data-toc-open
    // synchronously; doing that before any layout read means the very first
    // style computation for this freshly-inserted node already reflects the
    // resting state, so a later attribute flip can never be a change from a
    // real prior frame that .toc-panel's 0.22s transition would animate on
    // page load. This mattered acutely while a pinned mode existed (the panel
    // opened at load, so the flip was visible); with the panel now always
    // starting closed the ordering is cheap insurance rather than a live bug,
    // and it stays because reintroducing any open-at-load state would bring
    // the flash straight back, silently — ordinary test timing does not
    // reliably catch it.
    instance.reveal = createReveal(instance, doc);

    var rails = renderTicks(shell.rail, entries, doc);
    instance.ticks = rails;

    instance.accordion = createAccordion(instance.listEl);

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

        // Only follow the reader's position while no filter is active. apply()
        // force-expands every branch so deep matches are visible; syncTo()
        // would immediately collapse everything outside the reader's own
        // ancestor chain and take matched rows down with it. That fires from
        // any scroll while the panel is open, and even with no scrolling at
        // all when MathJax reflow re-measures.
        // filter.clear() re-syncs against the reader's real position, so the
        // accordion catches up the moment the filter is dropped.
        if (!(instance.filter && instance.filter.active)) instance.accordion.syncTo(id, topId);

        // Public hook, not dead code: no listener ships with the site; the
        // event exists so page-level scripts can follow the active heading
        // without reaching into instance internals.
        doc.dispatchEvent(new CustomEvent('toc:active', { detail: { id: id, topId: topId } }));
      },
      rails.position
    );

    // Ordering constraint: createFilter must be wired before createKeyboard.
    // Both attach an 'input' listener to the same filter field, and listeners
    // fire in registration order. createFilter's marks the rows in or out;
    // createKeyboard's then invalidates its row cache and calls highlight(0).
    // Swapped, highlight(0) runs against the *previous* keystroke's visible
    // set and outlines a row the query no longer matches. Nothing throws, so
    // this fails silently — keep filter first.
    instance.filter = createFilter(instance, doc);
    instance.keyboard = createKeyboard(instance, doc);

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
