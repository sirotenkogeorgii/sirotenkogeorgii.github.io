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

  window.TocPanel = {
    MIN_ENTRIES: MIN_ENTRIES,
    stripMath: stripMath,
    collectEntries: collectEntries,
    buildTree: buildTree,
    shouldActivate: shouldActivate
  };
})();
