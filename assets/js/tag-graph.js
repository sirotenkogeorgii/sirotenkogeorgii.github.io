/* Tag graph: force-directed view of tag co-occurrence across the site.
 * Data comes from /assets/data/tag-graph.json (built by Jekyll); nodes are
 * tags, an edge means two tags share at least one page, edge weight = number
 * of shared pages. Rendering is Canvas 2D via d3-force / d3-zoom / d3-drag. */
(function () {
  'use strict';

  // ---------------------------------------------------------------------
  // Pure graph construction (kept DOM-free so it can be smoke-tested in Node)
  // ---------------------------------------------------------------------

  function slugify(text) {
    return String(text)
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }

  function buildGraph(pages) {
    var tagInfo = new Map();
    var pairWeights = new Map();

    pages.forEach(function (page, pageIndex) {
      var unique = Array.from(new Set(page.tags || []))
        .filter(Boolean)
        .map(String)
        .sort();

      unique.forEach(function (tag) {
        if (!tagInfo.has(tag)) tagInfo.set(tag, { count: 0, pages: [] });
        var info = tagInfo.get(tag);
        info.count += 1;
        info.pages.push(pageIndex);
      });

      for (var i = 0; i < unique.length; i += 1) {
        for (var j = i + 1; j < unique.length; j += 1) {
          var key = unique[i] + '\u0000' + unique[j];
          pairWeights.set(key, (pairWeights.get(key) || 0) + 1);
        }
      }
    });

    var nodes = Array.from(tagInfo.keys()).sort().map(function (tag) {
      var info = tagInfo.get(tag);
      return { id: tag, count: info.count, pages: info.pages };
    });

    var links = Array.from(pairWeights.keys()).sort().map(function (key) {
      var parts = key.split('\u0000');
      return { source: parts[0], target: parts[1], weight: pairWeights.get(key) };
    });

    return { nodes: nodes, links: links };
  }

  /* Deterministic label propagation; returns Map(tag -> community rank),
   * rank 0 being the largest community. Used only for node colors. */
  function labelCommunities(nodes, links) {
    var labels = new Map();
    var neighbors = new Map();
    nodes.forEach(function (n) {
      labels.set(n.id, n.id);
      neighbors.set(n.id, []);
    });
    links.forEach(function (l) {
      var s = typeof l.source === 'object' ? l.source.id : l.source;
      var t = typeof l.target === 'object' ? l.target.id : l.target;
      neighbors.get(s).push({ id: t, weight: l.weight });
      neighbors.get(t).push({ id: s, weight: l.weight });
    });

    var order = nodes.map(function (n) { return n.id; }).sort();
    for (var iter = 0; iter < 12; iter += 1) {
      var changed = false;
      order.forEach(function (id) {
        var tally = new Map();
        neighbors.get(id).forEach(function (nb) {
          var lbl = labels.get(nb.id);
          tally.set(lbl, (tally.get(lbl) || 0) + nb.weight);
        });
        if (!tally.size) return;
        var best = labels.get(id);
        var bestScore = -1;
        Array.from(tally.keys()).sort().forEach(function (lbl) {
          var score = tally.get(lbl);
          if (score > bestScore) { bestScore = score; best = lbl; }
        });
        if (best !== labels.get(id)) {
          labels.set(id, best);
          changed = true;
        }
      });
      if (!changed) break;
    }

    var groups = new Map();
    labels.forEach(function (lbl, id) {
      if (!groups.has(lbl)) groups.set(lbl, []);
      groups.get(lbl).push(id);
    });
    var ranked = Array.from(groups.keys()).sort(function (a, b) {
      var diff = groups.get(b).length - groups.get(a).length;
      if (diff !== 0) return diff;
      return a < b ? -1 : 1;
    });
    var result = new Map();
    ranked.forEach(function (lbl, rank) {
      groups.get(lbl).forEach(function (id) { result.set(id, rank); });
    });
    return result;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { buildGraph: buildGraph, labelCommunities: labelCommunities, slugify: slugify };
    return;
  }
  if (typeof document === 'undefined' || typeof window === 'undefined') return;

  // ---------------------------------------------------------------------
  // Browser app
  // ---------------------------------------------------------------------

  var PALETTE = [
    '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f',
    '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#86bcb6',
    '#8cd17d', '#d4a6c8'
  ];
  var TAU = Math.PI * 2;

  function nodeRadius(count) {
    return 3.5 + 2.2 * Math.sqrt(count);
  }

  function edgeWidth(weight) {
    return Math.min(3.2, 0.5 + 0.45 * Math.sqrt(weight));
  }

  function edgeAlpha(weight, dark) {
    var base = dark ? 0.16 : 0.12;
    return Math.min(0.5, base + 0.07 * Math.sqrt(weight - 1));
  }

  function init() {
    var root = document.getElementById('tag-graph');
    if (!root || typeof d3 === 'undefined') {
      if (root) showMessage(root, 'The graph library failed to load.');
      return;
    }

    var stage = document.getElementById('tag-graph-stage');
    var canvas = document.getElementById('tag-graph-canvas');
    var tooltip = document.getElementById('tag-graph-tooltip');
    var panel = document.getElementById('tag-graph-panel');
    var message = document.getElementById('tag-graph-message');
    var searchInput = document.getElementById('tag-graph-search');
    var searchResults = document.getElementById('tag-graph-search-results');
    var thresholdInput = document.getElementById('tag-graph-threshold');
    var thresholdValue = document.getElementById('tag-graph-threshold-value');
    var statsEl = document.getElementById('tag-graph-stats');
    var resetButton = document.getElementById('tag-graph-reset');
    var tagsUrl = root.dataset.tagsUrl || '/tags/';

    var ctx = canvas.getContext('2d');
    var motionReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var bodyFont = getComputedStyle(document.body).fontFamily || 'sans-serif';

    var state = {
      pages: [],
      nodes: [],
      links: [],
      filteredLinks: [],
      adjacency: new Map(),        // full graph: id -> [{node, weight}] sorted by weight desc
      visibleNeighbors: new Map(), // filtered graph: id -> Set(ids), includes self
      minWeight: 1,
      hover: null,
      selected: null,
      transform: d3.zoomIdentity,
      width: 0,
      height: 0,
      theme: readTheme()
    };

    var sim = null;
    var zoom = null;
    var rafHandle = null;

    function readTheme() {
      var styles = getComputedStyle(document.documentElement);
      var dark = document.documentElement.dataset.theme === 'dark';
      function cssVar(name, fallback) {
        var value = styles.getPropertyValue(name).trim();
        return value || fallback;
      }
      return {
        dark: dark,
        text: cssVar('--text', dark ? '#e7ecf5' : '#1f2430'),
        muted: cssVar('--muted', dark ? '#9aa7bd' : '#5b6472'),
        surface: cssVar('--surface', dark ? '#0f172a' : '#ffffff'),
        accent: cssVar('--accent', dark ? '#60a5fa' : '#1d4ed8'),
        line: dark ? '#94a3b8' : '#64748b'
      };
    }

    function showMessage(target, text) {
      var el = target.querySelector('.tag-graph__message') || message;
      if (el) {
        el.hidden = false;
        el.textContent = text;
      }
    }

    function schedulePaint() {
      if (rafHandle) return;
      rafHandle = window.requestAnimationFrame(function () {
        rafHandle = null;
        paint();
      });
    }

    // ------------------------------------------------------------------
    // Painting
    // ------------------------------------------------------------------

    function paint() {
      var t = state.transform;
      var theme = state.theme;
      ctx.save();
      ctx.clearRect(0, 0, state.width, state.height);
      ctx.translate(t.x, t.y);
      ctx.scale(t.k, t.k);

      var focus = state.hover || state.selected;
      var focusSet = focus ? state.visibleNeighbors.get(focus.id) : null;

      ctx.lineCap = 'round';
      state.filteredLinks.forEach(function (link) {
        var active = focus && (link.source.id === focus.id || link.target.id === focus.id);
        if (focusSet && !active) {
          ctx.globalAlpha = 0.03;
        } else if (active) {
          ctx.globalAlpha = 0.85;
        } else {
          ctx.globalAlpha = edgeAlpha(link.weight, theme.dark);
        }
        ctx.strokeStyle = active ? theme.accent : theme.line;
        ctx.lineWidth = edgeWidth(link.weight) * (active ? 1.25 : 1);
        ctx.beginPath();
        ctx.moveTo(link.source.x, link.source.y);
        ctx.lineTo(link.target.x, link.target.y);
        ctx.stroke();
      });

      state.nodes.forEach(function (node) {
        var dimmed = focusSet && !focusSet.has(node.id);
        var isolated = node.visibleDegree === 0 && state.minWeight > 1;
        ctx.globalAlpha = dimmed ? 0.10 : (isolated ? 0.30 : 1);
        ctx.fillStyle = PALETTE[node.community % PALETTE.length];
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r, 0, TAU);
        ctx.fill();

        if (node === state.selected || node === state.hover) {
          ctx.globalAlpha = 1;
          ctx.strokeStyle = theme.accent;
          ctx.lineWidth = Math.max(1.5, 2.4 / t.k);
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r + Math.max(1.5, 2.4 / t.k), 0, TAU);
          ctx.stroke();
        }
      });

      ctx.restore();
      paintLabels(focus, focusSet);
    }

    function paintLabels(focus, focusSet) {
      var t = state.transform;
      var theme = state.theme;
      ctx.save();
      ctx.font = '600 12px ' + bodyFont;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.lineJoin = 'round';

      state.nodes.forEach(function (node) {
        var show;
        if (focusSet) {
          show = focusSet.has(node.id);
        } else {
          show = node.r * t.k >= 10;
        }
        if (!show) return;

        var sx = t.applyX(node.x);
        var sy = t.applyY(node.y);
        if (sx < -80 || sx > state.width + 80 || sy < -40 || sy > state.height + 40) return;

        ctx.globalAlpha = focusSet && focus !== node && state.hover !== node ? 0.85 : 1;
        var y = sy + node.r * t.k + 3;
        ctx.strokeStyle = theme.surface;
        ctx.lineWidth = 3;
        ctx.strokeText(node.id, sx, y);
        ctx.fillStyle = theme.text;
        ctx.fillText(node.id, sx, y);
      });
      ctx.restore();
    }

    // ------------------------------------------------------------------
    // Geometry helpers
    // ------------------------------------------------------------------

    function findNodeAtEvent(event) {
      var p = d3.pointer(event, canvas);
      var g = state.transform.invert(p);
      var candidate = sim.find(g[0], g[1], 30 / Math.min(1, state.transform.k) + 24);
      if (!candidate) return null;
      var dx = candidate.x - g[0];
      var dy = candidate.y - g[1];
      var slack = Math.max(2, 5 / state.transform.k);
      if (Math.sqrt(dx * dx + dy * dy) > candidate.r + slack) return null;
      return candidate;
    }

    function fitView(animate) {
      if (!state.nodes.length) return;
      var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
      state.nodes.forEach(function (n) {
        xMin = Math.min(xMin, n.x - n.r);
        xMax = Math.max(xMax, n.x + n.r);
        yMin = Math.min(yMin, n.y - n.r);
        yMax = Math.max(yMax, n.y + n.r);
      });
      var pad = 48;
      var k = Math.min(
        2,
        (state.width - pad * 2) / Math.max(1, xMax - xMin),
        (state.height - pad * 2) / Math.max(1, yMax - yMin)
      );
      k = Math.max(0.05, k);
      var cx = (xMin + xMax) / 2;
      var cy = (yMin + yMax) / 2;
      var t = d3.zoomIdentity
        .translate(state.width / 2, state.height / 2)
        .scale(k)
        .translate(-cx, -cy);
      applyTransform(t, animate);
    }

    function flyToNode(node) {
      var k = Math.max(1.3, state.transform.k);
      var t = d3.zoomIdentity
        .translate(state.width / 2, state.height / 2)
        .scale(k)
        .translate(-node.x, -node.y);
      applyTransform(t, true);
    }

    function applyTransform(t, animate) {
      var selection = d3.select(canvas);
      if (animate && !motionReduced) {
        selection.transition().duration(600).call(zoom.transform, t);
      } else {
        selection.call(zoom.transform, t);
      }
    }

    // ------------------------------------------------------------------
    // Graph state
    // ------------------------------------------------------------------

    function applyThreshold(minWeight, reheat) {
      state.minWeight = minWeight;
      state.filteredLinks = state.links.filter(function (l) { return l.weight >= minWeight; });

      var neighborSets = new Map();
      state.nodes.forEach(function (n) {
        n.visibleDegree = 0;
        neighborSets.set(n.id, new Set([n.id]));
      });
      state.filteredLinks.forEach(function (l) {
        var s = typeof l.source === 'object' ? l.source : null;
        var t = typeof l.target === 'object' ? l.target : null;
        var sid = s ? s.id : l.source;
        var tid = t ? t.id : l.target;
        neighborSets.get(sid).add(tid);
        neighborSets.get(tid).add(sid);
      });
      state.visibleNeighbors = neighborSets;
      state.nodes.forEach(function (n) {
        n.visibleDegree = neighborSets.get(n.id).size - 1;
      });

      sim.force('link').links(state.filteredLinks);
      if (reheat) sim.alpha(0.45).restart();

      var plural = state.filteredLinks.length === 1 ? '' : 's';
      statsEl.textContent = state.nodes.length + ' tags · ' + state.filteredLinks.length +
        ' connection' + plural + (minWeight > 1 ? ' (≥' + minWeight + ' shared pages)' : '');
    }

    function selectNode(node, fly) {
      state.selected = node;
      renderPanel(node);
      var params = new URLSearchParams(window.location.search);
      if (node) params.set('tag', node.id); else params.delete('tag');
      var query = params.toString();
      window.history.replaceState(null, '', window.location.pathname + (query ? '?' + query : ''));
      if (node && fly) flyToNode(node);
      schedulePaint();
    }

    // ------------------------------------------------------------------
    // Panel
    // ------------------------------------------------------------------

    function el(tag, className, text) {
      var node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function renderPanel(node) {
      panel.innerHTML = '';
      if (!node) {
        panel.hidden = true;
        return;
      }
      panel.hidden = false;

      var header = el('div', 'tag-graph__panel-header');
      var title = el('h2', 'tag-graph__panel-title', node.id);
      var close = el('button', 'tag-graph__panel-close', '×');
      close.type = 'button';
      close.setAttribute('aria-label', 'Close panel');
      close.addEventListener('click', function () { selectNode(null); });
      header.appendChild(title);
      header.appendChild(close);
      panel.appendChild(header);

      var meta = el('p', 'tag-graph__panel-meta');
      meta.textContent = node.count + (node.count === 1 ? ' page · ' : ' pages · ') +
        (state.adjacency.get(node.id) || []).length + ' related tags';
      panel.appendChild(meta);

      var indexLink = el('a', 'tag-graph__panel-link', 'Open in tag index ↗');
      indexLink.href = tagsUrl + '#tag-' + slugify(node.id);
      panel.appendChild(indexLink);

      var related = state.adjacency.get(node.id) || [];
      if (related.length) {
        panel.appendChild(el('h3', 'tag-graph__panel-subtitle', 'Related tags'));
        var chips = el('div', 'tag-graph__panel-chips');
        related.slice(0, 14).forEach(function (entry) {
          var chip = el('button', 'tag-graph__chip');
          chip.type = 'button';
          chip.appendChild(el('span', null, entry.node.id));
          chip.appendChild(el('span', 'tag-graph__chip-count', String(entry.weight)));
          chip.addEventListener('click', function () { selectNode(entry.node, true); });
          chips.appendChild(chip);
        });
        panel.appendChild(chips);
      }

      panel.appendChild(el('h3', 'tag-graph__panel-subtitle', 'Pages'));
      var list = el('ul', 'tag-graph__panel-pages');
      node.pages
        .map(function (i) { return state.pages[i]; })
        .sort(function (a, b) { return a.title.localeCompare(b.title); })
        .forEach(function (page) {
          var item = el('li');
          var link = el('a', null, page.title);
          link.href = page.url;
          item.appendChild(link);
          list.appendChild(item);
        });
      panel.appendChild(list);
      panel.scrollTop = 0;
    }

    // ------------------------------------------------------------------
    // Tooltip
    // ------------------------------------------------------------------

    function updateTooltip(node, clientX, clientY) {
      if (!node) {
        tooltip.hidden = true;
        return;
      }
      tooltip.textContent = node.id + ' — ' + node.count + (node.count === 1 ? ' page' : ' pages');
      tooltip.hidden = false;
      var stageRect = stage.getBoundingClientRect();
      var x = clientX - stageRect.left + 14;
      var y = clientY - stageRect.top + 14;
      var maxX = stageRect.width - tooltip.offsetWidth - 8;
      var maxY = stageRect.height - tooltip.offsetHeight - 8;
      tooltip.style.left = Math.min(x, Math.max(8, maxX)) + 'px';
      tooltip.style.top = Math.min(y, Math.max(8, maxY)) + 'px';
    }

    // ------------------------------------------------------------------
    // Search
    // ------------------------------------------------------------------

    var searchActiveIndex = -1;

    function renderSearchResults(query) {
      searchResults.innerHTML = '';
      searchActiveIndex = -1;
      var q = query.trim().toLowerCase();
      if (!q) {
        searchResults.hidden = true;
        searchInput.setAttribute('aria-expanded', 'false');
        return;
      }
      var matches = state.nodes
        .filter(function (n) { return n.id.toLowerCase().indexOf(q) !== -1; })
        .sort(function (a, b) {
          var aStarts = a.id.toLowerCase().indexOf(q) === 0 ? 0 : 1;
          var bStarts = b.id.toLowerCase().indexOf(q) === 0 ? 0 : 1;
          if (aStarts !== bStarts) return aStarts - bStarts;
          return b.count - a.count;
        })
        .slice(0, 12);

      if (!matches.length) {
        var empty = el('li', 'tag-graph__search-empty', 'No matching tags');
        searchResults.appendChild(empty);
      } else {
        matches.forEach(function (node) {
          var item = el('li');
          item.setAttribute('role', 'option');
          var button = el('button', 'tag-graph__search-option');
          button.type = 'button';
          button.appendChild(el('span', null, node.id));
          button.appendChild(el('span', 'tag-graph__chip-count', String(node.count)));
          button.addEventListener('click', function () { chooseSearchResult(node); });
          item.appendChild(button);
          searchResults.appendChild(item);
        });
      }
      searchResults.hidden = false;
      searchInput.setAttribute('aria-expanded', 'true');
    }

    function chooseSearchResult(node) {
      searchInput.value = node.id;
      searchResults.hidden = true;
      searchInput.setAttribute('aria-expanded', 'false');
      selectNode(node, true);
    }

    function moveSearchSelection(delta) {
      var options = Array.from(searchResults.querySelectorAll('.tag-graph__search-option'));
      if (!options.length) return;
      searchActiveIndex = (searchActiveIndex + delta + options.length) % options.length;
      options.forEach(function (option, index) {
        option.classList.toggle('is-active', index === searchActiveIndex);
      });
    }

    function bindSearch() {
      searchInput.addEventListener('input', function () {
        renderSearchResults(searchInput.value);
      });
      searchInput.addEventListener('keydown', function (event) {
        if (event.key === 'ArrowDown') {
          event.preventDefault();
          moveSearchSelection(1);
        } else if (event.key === 'ArrowUp') {
          event.preventDefault();
          moveSearchSelection(-1);
        } else if (event.key === 'Enter') {
          event.preventDefault();
          var options = Array.from(searchResults.querySelectorAll('.tag-graph__search-option'));
          var target = options[searchActiveIndex >= 0 ? searchActiveIndex : 0];
          if (target) target.click();
        } else if (event.key === 'Escape') {
          searchResults.hidden = true;
          searchInput.setAttribute('aria-expanded', 'false');
        }
      });
      document.addEventListener('click', function (event) {
        if (!searchResults.hidden && !searchResults.contains(event.target) && event.target !== searchInput) {
          searchResults.hidden = true;
          searchInput.setAttribute('aria-expanded', 'false');
        }
      });
    }

    // ------------------------------------------------------------------
    // Canvas sizing
    // ------------------------------------------------------------------

    function resizeCanvas() {
      var rect = stage.getBoundingClientRect();
      var dpr = window.devicePixelRatio || 1;
      state.width = Math.max(1, Math.round(rect.width));
      state.height = Math.max(1, Math.round(rect.height));
      canvas.width = state.width * dpr;
      canvas.height = state.height * dpr;
      canvas.style.width = state.width + 'px';
      canvas.style.height = state.height + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      schedulePaint();
    }

    // ------------------------------------------------------------------
    // Boot
    // ------------------------------------------------------------------

    function boot(data) {
      state.pages = data.pages || [];
      if (!state.pages.length) {
        showMessage(root, 'No tagged pages found yet.');
        return;
      }

      var graph = buildGraph(state.pages);
      state.nodes = graph.nodes;
      state.links = graph.links;

      var communities = labelCommunities(state.nodes, state.links);
      state.nodes.forEach(function (n) {
        n.r = nodeRadius(n.count);
        n.community = communities.get(n.id) || 0;
      });

      // Full-graph adjacency for the panel (independent of the threshold filter).
      state.adjacency = new Map();
      state.nodes.forEach(function (n) { state.adjacency.set(n.id, []); });
      var byId = new Map(state.nodes.map(function (n) { return [n.id, n]; }));
      state.links.forEach(function (l) {
        state.adjacency.get(l.source).push({ node: byId.get(l.target), weight: l.weight });
        state.adjacency.get(l.target).push({ node: byId.get(l.source), weight: l.weight });
      });
      state.adjacency.forEach(function (list) {
        list.sort(function (a, b) { return b.weight - a.weight || a.node.id.localeCompare(b.node.id); });
      });

      sim = d3.forceSimulation(state.nodes)
        .force('link', d3.forceLink()
          .id(function (d) { return d.id; })
          .distance(function (l) {
            return 24 + (l.source.r + l.target.r) - Math.min(14, l.weight * 1.6);
          }))
        .force('charge', d3.forceManyBody().strength(-90).distanceMax(420))
        .force('collide', d3.forceCollide(function (d) { return d.r + 3; }).strength(0.9))
        .force('x', d3.forceX(0).strength(0.05))
        .force('y', d3.forceY(0).strength(0.05))
        .stop();

      var maxWeight = state.links.reduce(function (m, l) { return Math.max(m, l.weight); }, 1);
      thresholdInput.max = String(Math.max(2, Math.min(10, maxWeight)));
      applyThreshold(1, false);

      // Settle off-screen so the page never shows the initial explosion.
      var preTicks = motionReduced ? 320 : 160;
      for (var i = 0; i < preTicks; i += 1) sim.tick();

      zoom = d3.zoom()
        .scaleExtent([0.05, 8])
        .filter(function (event) {
          if (event.type === 'wheel' || event.type === 'dblclick') return true;
          if (event.button) return false;
          if (event.touches && event.touches.length > 1) return true;
          return !findNodeAtEvent(event);
        })
        .on('zoom', function (event) {
          state.transform = event.transform;
          schedulePaint();
        });

      var drag = d3.drag()
        .container(canvas)
        .subject(function (event) {
          return findNodeAtEvent(event.sourceEvent || event);
        })
        .on('start', function (event) {
          sim.alphaTarget(0.25).restart();
          event.subject.fx = event.subject.x;
          event.subject.fy = event.subject.y;
          canvas.classList.add('is-dragging');
        })
        .on('drag', function (event) {
          var p = state.transform.invert(d3.pointer(event, canvas));
          event.subject.fx = p[0];
          event.subject.fy = p[1];
          schedulePaint();
        })
        .on('end', function (event) {
          sim.alphaTarget(0);
          event.subject.fx = null;
          event.subject.fy = null;
          canvas.classList.remove('is-dragging');
        });

      d3.select(canvas).call(drag).call(zoom);

      canvas.addEventListener('mousemove', function (event) {
        var node = findNodeAtEvent(event);
        if (node !== state.hover) {
          state.hover = node;
          schedulePaint();
        }
        canvas.style.cursor = node ? 'pointer' : '';
        updateTooltip(node, event.clientX, event.clientY);
      });
      canvas.addEventListener('mouseleave', function () {
        if (state.hover) {
          state.hover = null;
          schedulePaint();
        }
        updateTooltip(null);
      });
      canvas.addEventListener('click', function (event) {
        if (event.defaultPrevented) return;
        selectNode(findNodeAtEvent(event) || null);
      });
      document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && state.selected && document.activeElement !== searchInput) {
          selectNode(null);
        }
      });

      sim.on('tick', schedulePaint);
      if (!motionReduced) sim.alpha(0.18).restart();

      thresholdInput.addEventListener('input', function () {
        var value = Number(thresholdInput.value) || 1;
        thresholdValue.textContent = String(value);
        applyThreshold(value, true);
        schedulePaint();
      });
      resetButton.addEventListener('click', function () { fitView(true); });
      bindSearch();

      var observer = new MutationObserver(function () {
        state.theme = readTheme();
        schedulePaint();
      });
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

      if ('ResizeObserver' in window) {
        new ResizeObserver(resizeCanvas).observe(stage);
      }
      window.addEventListener('resize', resizeCanvas);

      message.hidden = true;
      resizeCanvas();
      fitView(false);

      var preselect = new URLSearchParams(window.location.search).get('tag');
      if (preselect) {
        var match = byId.get(preselect) || byId.get(slugify(preselect)) ||
          state.nodes.find(function (n) { return slugify(n.id) === slugify(preselect); });
        if (match) selectNode(match, true);
      }
    }

    fetch(root.dataset.source)
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(boot)
      .catch(function (error) {
        showMessage(root, 'Could not load the tag data (' + error.message + '). ' +
          'The tag index still works as a fallback.');
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
