# TOC Navigation Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unstyled inline table of contents on 64 long pages with a hover-revealed right-edge panel that collapses to the section being read, filters, and works by keyboard.

**Architecture:** Pure client-side progressive enhancement. Two new vanilla-JS files enhance the `<ul id="markdown-toc">` that kramdown already emits, then reparent it into a fixed-position panel. If the scripts do not run, pages render exactly as they do today. No markdown files change; no Jekyll plugins are added.

**Tech Stack:** Jekyll 4.3 (local preview via `Gemfile.local`), kramdown/GFM, MathJax 3, vanilla ES5-style JS, plain CSS custom properties. Tests: Node 24 built-in test runner (`node --test`) driving Playwright 1.62 headless Chromium against fixtures served by `python3 -m http.server`.

## Global Constraints

- **No markdown files may be edited.** All 64 TOC pages use `layout: default`; that layout is the only integration point.
- **No new Jekyll plugins.** GitHub Pages rejects them.
- **No runtime JS dependencies.** The shipped files must be plain `<script>`-loadable, no bundler, no imports.
- **Test tooling is gitignored.** `package.json`, `package-lock.json`, `node_modules/` and `tests/` must never be committed and must never ship to `_site`.
- Panel width `280px`; pin breakpoint `min-width: 1660px`; mobile breakpoint `max-width: 899px` (pairs with the existing `900px` breakpoint in `site.css`).
- Activation threshold: the panel initialises only when the page has a `ul#markdown-toc` with **4 or more** entries.
- Styling uses only existing custom properties: `--surface`, `--surface-muted`, `--line`, `--muted`, `--accent`, `--accent-muted`, `--shadow`, `--font-body`.
- `.site-header` is `z-index: 30`; the panel uses `z-index: 40`.
- Reduced motion, print, and dark theme must all be handled (Task 11).

## Deviations from the spec

Five, all deliberate. Record them in the commit messages.

1. **Two JS files, not one.** `createAnchors()` is described in the spec as "fully independent of the panel; deleting it touches nothing else." It therefore ships as `assets/js/heading-anchors.js` rather than being bolted into `toc-panel.js`, which keeps the panel file under ~400 lines.
2. **Scroll spy uses a rAF-throttled scroll handler, not `IntersectionObserver`.** The spec proposed `rootMargin: '0px 0px -80% 0px'`. That band is 20vh tall, but sections in these notes are routinely several screens tall, so no heading intersects the band for most of the scroll and the active item would blank out. A cached-offsets + binary-scan approach is correct for tall sections and is what a `ResizeObserver` can keep accurate as MathJax reflows the page.
3. **Automated tests exist.** The spec said manual verification only. Approved change: gitignored Playwright harness.
4. **`createReveal` is wired before `renderTicks`/`createAccordion`/`createScrollSpy` in `init`, not after them as Task 6 originally specified.** `renderTicks` and `createScrollSpy` both call `getBoundingClientRect()`, which forces a synchronous layout pass and commits `.toc-panel`'s off-screen `translateX(100%)` as an already-rendered style. If that layout happened first, `createReveal`'s pinned-mode attribute flip would be a *change* from a real prior frame and the panel's 0.22s transition would visibly slide it in on every page load. Wiring reveal first means the very first style computation for the freshly-inserted panel already bakes in the resting position, so there is nothing to transition from. Reviewed and judged sound; do not "align this with the brief" — that silently reintroduces the on-load flash, since ordinary test timing does not reliably catch it.
5. **The `@media print` block sets `transition: none` on `.toc-panel`.** Task 6's Critical finding (see Task 6 below) was about suppressing a *screen* transition to satisfy a synchronous test assertion, which genuinely deleted a user-visible animation. This is categorically different: `@media print` cannot affect screen media at all, the protected hover-slide test never emulates print so it still exercises a real animation, and print is a static-snapshot context where nothing user-perceivable is being removed. Cleared as a normal, independently common print convention rather than a repeat of the Task 6 mistake.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `assets/js/toc-panel.js` | create | The panel: parse, decorate, render, scroll-spy, accordion, reveal, filter, keyboard |
| `assets/js/heading-anchors.js` | create | The `¶` copy-link affordance on `h1`–`h3`. Independent of the panel |
| `assets/css/site.css` | modify | Append one `Table-of-contents panel` section at the end (currently 1941 lines) |
| `_layouts/default.html` | modify | Two `<script defer>` tags in `<head>` |
| `.gitignore` | modify | Add `node_modules/`, `package.json`, `package-lock.json`, `tests/` |
| `_config.yml` | modify | Add the same paths to `exclude:` |
| `package.json` | create (gitignored) | Playwright devDependency + `npm test` |
| `tests/helpers.mjs` | create (gitignored) | Server, browser, live-asset routing, fixture builder |
| `tests/*.spec.mjs` | create (gitignored) | One spec file per task |

### Why the test loop is fast

Tests do **not** require a Jekyll build. `tests/helpers.mjs` serves the **repository root** over HTTP and writes fixture pages into `tests/fixtures/`, so a fixture at `/tests/fixtures/x.html` can reference `/assets/js/toc-panel.js` and `/assets/css/site.css` directly from the working tree. Edit JS, re-run tests, no rebuild.

A single real-page smoke test (Task 11) runs against `_site` and is skipped when `_site` is absent.

---

### Task 1: Test harness

**Files:**
- Create: `package.json`, `tests/helpers.mjs`, `tests/harness.spec.mjs`
- Modify: `.gitignore`, `_config.yml:56-63`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `startServer(): Promise<{origin: string, close: () => void}>` — serves repo root, ephemeral port.
  - `withPage(opts, fn): Promise<any>` — `opts = {viewport?, js?, origin}`; routes `/assets/**` to the working tree and blocks the MathJax CDN. Passes a Playwright `page` to `fn`.
  - `buildFixture(name, headings, opts?): string` — writes `tests/fixtures/<name>.html`, returns the path (`/tests/fixtures/<name>.html`). `headings` is `[{level, text}]`. `opts.tocEntries` may be set to `0` to emit a page with no TOC at all.

- [ ] **Step 1: Add the ignore entries**

Append to `.gitignore`:

```
node_modules/
package.json
package-lock.json
tests/
```

In `_config.yml`, the `exclude:` list currently replaces Jekyll's defaults — which means `node_modules` is **not** excluded unless we say so. Extend it to:

```yaml
exclude:
  - Gemfile
  - Gemfile.lock
  - README.md
  - vendor
  - tmp
  - "*.bak"
  - "**/*.bak"
  - node_modules
  - package.json
  - package-lock.json
  - tests
```

Do **not** add `docs` here. `docs/superpowers/` is currently published to the
site, and changing that is unrelated to this work.

- [ ] **Step 2: Create `package.json`**

```json
{
  "name": "personal-website-dev",
  "private": true,
  "type": "module",
  "scripts": {
    "test": "node --test --test-concurrency=1 \"tests/**/*.spec.mjs\"",
    "build": "BUNDLE_GEMFILE=Gemfile.local bundle exec jekyll build"
  },
  "devDependencies": {
    "playwright": "1.62.0"
  }
}
```

The glob is quoted and passed explicitly rather than the bare directory `tests/`: on Node 24, `node --test` does not recurse into a directory argument, so a bare `tests/` silently runs zero files.

- [ ] **Step 3: Install Playwright without re-downloading browsers**

`chromium_headless_shell-1228` is already in `~/Library/Caches/ms-playwright/`, so skip the download.

Run: `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install`
Expected: `added N packages`, no browser download.

If a later step fails with "Executable doesn't exist", run `npx playwright install chromium-headless-shell` once.

- [ ] **Step 4: Write the failing harness test**

Create `tests/harness.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture } from './helpers.mjs';

test('harness serves a fixture with the real stylesheet and script', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('harness', [
      { level: 1, text: 'Alpha' },
      { level: 2, text: 'Beta' }
    ]);
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await page.locator('#markdown-toc a').count(), 2);
      const bg = await page.evaluate(() =>
        getComputedStyle(document.body).backgroundColor);
      assert.notEqual(bg, 'rgba(0, 0, 0, 0)', 'site.css should have applied');
      assert.equal(await page.evaluate(() => typeof window.TocPanel), 'object');
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 5: Run it to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './helpers.mjs'`.

- [ ] **Step 6: Write `tests/helpers.mjs`**

```js
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIXTURE_DIR = path.join(REPO, 'tests', 'fixtures');

export async function startServer() {
  // -u: unbuffered stdout. Without it, Python's http.server buffers the
  // "Serving HTTP on ..." startup line (it isn't a tty on the other end of
  // the pipe) and we'd never see it in time. The line is also printed to
  // stdout, not stderr.
  const proc = spawn(
    'python3',
    ['-u', '-m', 'http.server', '0', '--bind', '127.0.0.1', '--directory', REPO],
    { stdio: ['ignore', 'pipe', 'ignore'] }
  );

  try {
    const port = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('server did not report a port')), 10000);
      proc.stdout.on('data', (chunk) => {
        const match = /port (\d+)/.exec(String(chunk));
        if (match) {
          clearTimeout(timer);
          resolve(Number(match[1]));
        }
      });
      proc.on('error', reject);
    });

    return {
      origin: `http://127.0.0.1:${port}`,
      close: () => proc.kill()
    };
  } catch (err) {
    // Never leave an orphaned http.server running if we didn't hand back a
    // close() the caller could use to kill it themselves.
    proc.kill();
    throw err;
  }
}

export async function withPage(opts, fn) {
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({
      viewport: opts.viewport ?? { width: 1280, height: 900 },
      javaScriptEnabled: opts.js !== false
    });
    const page = await context.newPage();

    // MathJax is slow and non-deterministic; block it so headings keep their raw
    // "$...$" source, which is exactly what the filter indexes.
    await page.route('**/cdn.jsdelivr.net/**', (route) => route.abort());

    // Serve JS and CSS from the working tree so no Jekyll rebuild is needed.
    await page.route('**/assets/js/*.js', (route) => {
      const name = path.basename(new URL(route.request().url()).pathname);
      const file = path.join(REPO, 'assets', 'js', name);
      return existsSync(file)
        ? route.fulfill({ path: file, contentType: 'application/javascript' })
        : route.continue();
    });
    await page.route('**/assets/css/site.css', (route) =>
      route.fulfill({ path: path.join(REPO, 'assets', 'css', 'site.css'), contentType: 'text/css' })
    );

    return await fn(page);
  } finally {
    await browser.close();
  }
}

// Mirrors what kramdown emits: a "<p><strong>Table of Contents</strong></p>"
// followed by a nested <ul id="markdown-toc">, then the headings themselves.
export function buildFixture(name, headings, opts = {}) {
  mkdirSync(FIXTURE_DIR, { recursive: true });

  const slug = (t, i) =>
    t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') + '-' + i;

  const entries = headings.map((h, i) => ({ ...h, id: slug(h.text, i) }));

  const toc = renderTocHtml(entries);

  const body = entries
    .map(
      (e) =>
        `<h${e.level} id="${e.id}">${e.text}</h${e.level}>` +
        `<p>${'Filler paragraph for scrolling. '.repeat(opts.filler ?? 40)}</p>`
    )
    .join('\n');

  const tocBlock =
    opts.tocEntries === 0
      ? ''
      : `<p><strong>Table of Contents</strong></p>\n${toc}`;

  const html = `<!DOCTYPE html>
<html lang="en" data-theme="light" data-font="default">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${name} fixture</title>
<link rel="stylesheet" href="/assets/css/site.css">
<script defer src="/assets/js/toc-panel.js"></script>
<script defer src="/assets/js/heading-anchors.js"></script>
</head>
<body class="layout-default">
<header class="site-header"><div class="wrapper"><a class="site-title" href="/">Fixture</a></div></header>
<main class="page-content"><div class="wrapper">
${tocBlock}
${body}
</div></main>
</body>
</html>`;

  writeFileSync(path.join(FIXTURE_DIR, name + '.html'), html);
  return `/tests/fixtures/${name}.html`;
}

// Nesting must be produced by walking levels, not by a flat loop that closes
// every <li> immediately — otherwise sublists end up as siblings.

function renderTocHtml(entries) {
  let out = '';
  let prev = 0;
  entries.forEach((e, i) => {
    if (e.level > prev) {
      for (let d = prev; d < e.level; d++) {
        out += d === 0 ? '<ul id="markdown-toc">' : '<ul>';
      }
    } else {
      out += '</li>';
      for (let d = prev; d > e.level; d--) out += '</ul></li>';
    }
    out += `<li><a href="#${e.id}">${e.text}</a>`;
    prev = e.level;
  });
  out += '</li>';
  for (let d = prev; d > 1; d--) out += '</ul></li>';
  out += '</ul>';
  return out;
}

export function headingSeries(count, perChapter = 12) {
  const out = [];
  for (let i = 0; i < count; i++) {
    if (i % perChapter === 0) out.push({ level: 1, text: `Chapter ${Math.floor(i / perChapter) + 1}` });
    else if (i % 3 === 0) out.push({ level: 2, text: `Section ${i}` });
    else out.push({ level: 3, text: `Topic ${i}` });
  }
  return out;
}
```

Delete the dead first-attempt `toc` loop before saving — `renderTocHtml` is the only builder used. It is left in the plan only to make clear that the nesting must be produced by the tree renderer, not by a flat loop.

- [ ] **Step 7: Create the placeholder script files so the fixture loads cleanly**

`assets/js/toc-panel.js`:

```js
(function () {
  'use strict';
  window.TocPanel = {};
})();
```

`assets/js/heading-anchors.js`:

```js
(function () {
  'use strict';
})();
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `npm test`
Expected: PASS, 1 test.

- [ ] **Step 9: Commit**

```bash
git add .gitignore _config.yml assets/js/toc-panel.js assets/js/heading-anchors.js
git commit -m "chore: add gitignored playwright harness and empty toc scripts"
```

Confirm the harness stayed out of the commit:

```bash
git status --short
```
Expected: `package.json`, `tests/`, `node_modules/` do **not** appear.

---

### Task 2: Parse the TOC and build the tree

**Files:**
- Modify: `assets/js/toc-panel.js`, `_layouts/default.html:28`
- Test: `tests/toc-model.spec.mjs`

**Interfaces:**
- Consumes: `buildFixture`, `withPage`, `startServer`, `headingSeries` from Task 1.
- Produces, on `window.TocPanel`:
  - `stripMath(text: string) => string`
  - `collectEntries(tocRoot: Element, doc: Document) => Entry[]` where `Entry = {id, level, text, filterKey, anchorEl, liEl, headingEl}`
  - `buildTree(entries: Entry[]) => Node[]` where `Node = {entry, children: Node[]}`
  - `shouldActivate(entries: Entry[]) => boolean`
  - `MIN_ENTRIES = 4`

- [ ] **Step 1: Write the failing test**

Create `tests/toc-model.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture } from './helpers.mjs';

const HEADINGS = [
  { level: 1, text: 'Chapter 1: Introduction' },
  { level: 2, text: 'Sequential Decision Making' },
  { level: 3, text: 'The $\\ell^0$ Norm' },
  { level: 2, text: 'Canonical Models' },
  { level: 1, text: 'Chapter 2: Value-based RL' }
];

test('collectEntries reads ids, levels and text', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('model', HEADINGS);
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      const entries = await page.evaluate(() => {
        const root = document.getElementById('markdown-toc');
        return window.TocPanel.collectEntries(root, document)
          .map((e) => ({ id: e.id, level: e.level, text: e.text, filterKey: e.filterKey }));
      });

      assert.equal(entries.length, 5);
      assert.deepEqual(entries.map((e) => e.level), [1, 2, 3, 2, 1]);
      assert.equal(entries[0].id, 'chapter-1-introduction-0');
      assert.equal(entries[2].text, 'The $\\ell^0$ Norm');
    });
  } finally {
    server.close();
  }
});

test('stripMath makes math headings searchable', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('model-math', HEADINGS);
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      const keys = await page.evaluate(() => ({
        norm: window.TocPanel.stripMath('2.2.2 The $\\ell^0$ Norm'),
        value: window.TocPanel.stripMath('Solution for $V^\\pi$'),
        plain: window.TocPanel.stripMath('  Canonical  Models ')
      }));

      assert.match(keys.norm, /norm/);
      assert.match(keys.norm, /2\.2\.2/);
      assert.doesNotMatch(keys.norm, /\$/);
      assert.doesNotMatch(keys.value, /\\pi/);
      assert.match(keys.value, /solution for/);
      assert.equal(keys.plain, 'canonical models');
    });
  } finally {
    server.close();
  }
});

test('buildTree nests by level', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('model-tree', HEADINGS);
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      const shape = await page.evaluate(() => {
        const root = document.getElementById('markdown-toc');
        const tree = window.TocPanel.buildTree(window.TocPanel.collectEntries(root, document));
        const walk = (n) => ({ text: n.entry.text, children: n.children.map(walk) });
        return tree.map(walk);
      });

      assert.equal(shape.length, 2, 'two top-level chapters');
      assert.equal(shape[0].children.length, 2, 'chapter 1 has two H2s');
      assert.equal(shape[0].children[0].children.length, 1, 'first H2 has one H3');
      assert.equal(shape[1].children.length, 0);
    });
  } finally {
    server.close();
  }
});

test('shouldActivate is false below the threshold and for TOC-less pages', async () => {
  const server = await startServer();
  try {
    const small = buildFixture('model-small', HEADINGS.slice(0, 3));
    const none = buildFixture('model-none', HEADINGS, { tocEntries: 0 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + small);
      const activeSmall = await page.evaluate(() => {
        const root = document.getElementById('markdown-toc');
        return window.TocPanel.shouldActivate(window.TocPanel.collectEntries(root, document));
      });
      assert.equal(activeSmall, false);

      await page.goto(server.origin + none);
      assert.equal(await page.evaluate(() => !!document.getElementById('markdown-toc')), false);
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `window.TocPanel.collectEntries is not a function`.

- [ ] **Step 3: Implement the model layer**

Replace `assets/js/toc-panel.js` with:

```js
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `npm test`
Expected: PASS, 5 tests.

- [ ] **Step 5: Wire the scripts into the layout**

In `_layouts/default.html`, immediately after the stylesheet link on line 28, insert:

```html
  <script defer src="{{ '/assets/js/toc-panel.js' | relative_url }}"></script>
  <script defer src="{{ '/assets/js/heading-anchors.js' | relative_url }}"></script>
```

`defer` matters: deferred scripts run before `DOMContentLoaded`, and MathJax 3 typesets on page-ready, so the panel is built — and `filterKey` captured from the raw `$...$` source — before MathJax rewrites any heading text.

- [ ] **Step 6: Build the site once so `_site` has the new script tags**

Run: `BUNDLE_GEMFILE=Gemfile.local bundle exec jekyll build`
Expected: `done in N seconds`. This can take several minutes across 680 markdown files. It is needed **once**; from here on `tests/helpers.mjs` serves JS and CSS from the working tree.

- [ ] **Step 7: Commit**

```bash
git add assets/js/toc-panel.js _layouts/default.html
git commit -m "feat: parse markdown-toc into a heading model"
```

---

### Task 3: Render the panel and reparent the inline TOC

**Files:**
- Modify: `assets/js/toc-panel.js`, `assets/css/site.css` (append at end, currently line 1941)
- Test: `tests/toc-render.spec.mjs`

**Interfaces:**
- Consumes: `collectEntries`, `buildTree`, `shouldActivate` from Task 2.
- Produces, on `window.TocPanel`:
  - `init(doc?) => Instance | null`, auto-invoked on `DOMContentLoaded`. Returns `null` when the page should not activate.
  - `Instance = {entries, tree, root, hotzone, rail, panel, listEl, filterEl}` where `root` is `.toc-root` and `tree` is `buildTree(entries)` (kept on the instance as the model layer's shape contract, not read internally).
  - `window.TocPanel.instance` holds the live instance or `null`.
- DOM contract later tasks rely on: `.toc-root`, `.toc-root__hotzone`, `.toc-rail`, `.toc-panel`, `.toc-panel__filter`, `.toc-panel__list`, `li.toc-panel__item[data-toc-id]`, `li.toc-panel__item--branch[data-expanded]`, `button.toc-panel__twisty`, `a.toc-panel__link`.

- [ ] **Step 1: Write the failing test**

Create `tests/toc-render.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture, headingSeries } from './helpers.mjs';

test('the inline TOC is moved into the panel and its caption hidden', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('render', headingSeries(30));
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);

      assert.equal(
        await page.evaluate(() => !!document.querySelector('.toc-panel #markdown-toc')),
        true,
        'the ul should now live inside the panel'
      );
      assert.equal(
        await page.evaluate(() => !!document.querySelector('.page-content #markdown-toc')),
        false,
        'the ul should no longer be in the content flow'
      );

      const captionVisible = await page.evaluate(() => {
        const p = [...document.querySelectorAll('.page-content p strong')]
          .find((s) => s.textContent.trim() === 'Table of Contents');
        return p ? getComputedStyle(p.closest('p')).display !== 'none' : false;
      });
      assert.equal(captionVisible, false, '"Table of Contents" caption should be hidden');

      // Content now starts near the top of the document.
      const firstHeadingTop = await page.evaluate(
        () => document.querySelector('.page-content h1').getBoundingClientRect().top + window.scrollY
      );
      assert.ok(firstHeadingTop < 400, `first heading at ${firstHeadingTop}px, expected near top`);
    });
  } finally {
    server.close();
  }
});

test('branch items get a twisty and collapse by default', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('render-branches', headingSeries(30));
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      const counts = await page.evaluate(() => ({
        items: document.querySelectorAll('li.toc-panel__item[data-toc-id]').length,
        branches: document.querySelectorAll('li.toc-panel__item--branch').length,
        twisties: document.querySelectorAll('button.toc-panel__twisty').length
      }));
      assert.equal(counts.items, 30);
      assert.ok(counts.branches > 0);
      assert.equal(counts.twisties, counts.branches, 'one twisty per branch');
    });
  } finally {
    server.close();
  }
});

test('pages below threshold or without a TOC are untouched', async () => {
  const server = await startServer();
  try {
    const small = buildFixture('render-small', headingSeries(3, 2), { filler: 5 });
    const none = buildFixture('render-none', headingSeries(30), { tocEntries: 0 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + small);
      assert.equal(await page.locator('.toc-root').count(), 0);
      assert.equal(await page.evaluate(() => window.TocPanel.instance), null);

      await page.goto(server.origin + none);
      assert.equal(await page.locator('.toc-root').count(), 0);
    });
  } finally {
    server.close();
  }
});

test('with JS disabled the inline TOC survives', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('render-nojs', headingSeries(30));
    await withPage({ origin: server.origin, js: false }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await page.locator('.page-content #markdown-toc').count(), 1);
      assert.equal(await page.locator('.toc-root').count(), 0);
      const visible = await page.locator('.page-content #markdown-toc').isVisible();
      assert.equal(visible, true);
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `.toc-panel #markdown-toc` not found.

- [ ] **Step 3: Add decoration and rendering to `assets/js/toc-panel.js`**

Insert these functions before the `window.TocPanel = {...}` assignment:

```js
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
    return instance;
  }
```

Then extend the export object and auto-start:

```js
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
```

- [ ] **Step 4: Append the base CSS**

Append to the end of `assets/css/site.css`:

```css
/* ---------------------------------------------------------------
   Table-of-contents panel
   --------------------------------------------------------------- */

:root {
  --toc-panel-width: 280px;
  --toc-rail-width: 20px;
}

.toc-legacy-caption {
  display: none;
}

.toc-root {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  width: var(--toc-panel-width);
  z-index: 40;
  pointer-events: none;
  font-family: var(--font-body);
}

.toc-root__hotzone {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 40px;
  pointer-events: auto;
}

.toc-rail {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: var(--toc-rail-width);
  pointer-events: auto;
}

.toc-panel {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  pointer-events: auto;
  background: var(--surface);
  border-left: 1px solid var(--line);
  box-shadow: var(--shadow);
  transform: translateX(100%);
  transition: transform 0.22s ease;
}

.toc-panel__header {
  padding: 1rem 1rem 0.75rem;
  border-bottom: 1px solid var(--line);
}

.toc-panel__filter {
  width: 100%;
  font: inherit;
  font-size: 0.9rem;
  color: var(--text);
  background: var(--surface-muted);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.4rem 0.6rem;
}

.toc-panel__filter:focus {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}

.toc-panel__body {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem 0.5rem 2rem;
}

.toc-panel__list,
.toc-panel__sublist {
  list-style: none;
  margin: 0;
  padding: 0;
}

.toc-panel__sublist {
  margin-left: 0.85rem;
  padding-left: 0.6rem;
  border-left: 1px solid var(--line);
}

.toc-panel__item {
  position: relative;
  font-size: 0.86rem;
  line-height: 1.45;
}

.toc-panel__item--branch[data-expanded="false"] > .toc-panel__sublist {
  display: none;
}

.toc-panel__link {
  display: block;
  padding: 0.22rem 0.4rem 0.22rem 0.5rem;
  border-radius: 6px;
  color: var(--muted);
  border-left: 2px solid transparent;
}

.toc-panel__item--l1 > .toc-panel__link {
  color: var(--text);
  font-weight: 600;
}

.toc-panel__link:hover,
.toc-panel__link:focus {
  color: var(--accent);
  background: var(--accent-muted);
  text-decoration: none;
}

.toc-panel__twisty {
  position: absolute;
  left: -0.85rem;
  top: 0.35rem;
  width: 0.85rem;
  height: 0.85rem;
  padding: 0;
  border: 0;
  background: none;
  color: var(--muted);
  cursor: pointer;
  line-height: 1;
}

.toc-panel__twisty::before {
  content: "\25B8";
  display: block;
  font-size: 0.7rem;
  transition: transform 0.15s ease;
}

.toc-panel__item--branch[data-expanded="true"] > .toc-panel__twisty::before {
  transform: rotate(90deg);
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `npm test`
Expected: PASS, 13 tests.

- [ ] **Step 6: Commit**

```bash
git add assets/js/toc-panel.js assets/css/site.css
git commit -m "feat: reparent inline TOC into a fixed side panel"
```

---

### Task 4: Scroll spy and rail ticks

**Files:**
- Modify: `assets/js/toc-panel.js`, `assets/css/site.css`
- Test: `tests/toc-scrollspy.spec.mjs`

**Interfaces:**
- Consumes: `init` and the `Instance` shape from Task 3.
- Produces:
  - `createScrollSpy(entries, contentEl, onChange, onMeasure) => {measure(), refresh(), activeId}` — `onMeasure` is an optional callback invoked at the end of every `measure()` (initial load, resize, and `ResizeObserver` reflow), not from `onChange`. Step 4 below passes `rails.position` as `onMeasure` so tick placement stays correct after a MathJax reflow that involves no scrolling at all.
  - `instance.spy` on the instance.
  - DOM: `.toc-rail__tick[data-toc-id]`, gaining `.toc-rail__tick--active`; the active `li` gains `data-active="true"`.
  - Custom event `toc:active` dispatched on `document` with `detail.id`.

- [ ] **Step 1: Write the failing test**

Create `tests/toc-scrollspy.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture, headingSeries } from './helpers.mjs';

async function activeId(page) {
  return page.evaluate(() => {
    const el = document.querySelector('.toc-panel__item[data-active="true"]');
    return el ? el.getAttribute('data-toc-id') : null;
  });
}

test('the active heading follows the scroll position', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('spy', headingSeries(24), { filler: 60 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);

      const first = await activeId(page);
      assert.ok(first, 'something should be active at the top');

      const ids = await page.evaluate(() =>
        window.TocPanel.instance.entries.map((e) => e.id));
      const target = ids[12];

      await page.evaluate((id) => {
        const el = document.getElementById(id);
        window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 20);
      }, target);
      await page.waitForFunction(
        (id) => document.querySelector('.toc-panel__item[data-active="true"]')
          ?.getAttribute('data-toc-id') === id,
        target,
        { timeout: 4000 }
      );

      assert.equal(await activeId(page), target);
    });
  } finally {
    server.close();
  }
});

test('a section taller than the viewport keeps its heading active', async () => {
  const server = await startServer();
  try {
    // filler 400 makes each section far taller than one screen — the case that
    // an IntersectionObserver band would blank out.
    const url = buildFixture('spy-tall', headingSeries(8, 4), { filler: 400 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      const ids = await page.evaluate(() =>
        window.TocPanel.instance.entries.map((e) => e.id));

      await page.evaluate((id) => {
        const el = document.getElementById(id);
        window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY + 1500);
      }, ids[2]);
      await page.waitForTimeout(300);

      assert.equal(await activeId(page), ids[2], 'still inside section 2');
    });
  } finally {
    server.close();
  }
});

test('the rail draws one tick per top-level heading and marks the active branch', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('spy-ticks', headingSeries(36, 12), { filler: 60 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);

      const topLevel = await page.evaluate(() =>
        window.TocPanel.instance.entries.filter((e) => e.level === 1).length);
      assert.equal(await page.locator('.toc-rail__tick').count(), topLevel);

      const active = await page.locator('.toc-rail__tick--active').count();
      assert.equal(active, 1, 'exactly one tick highlighted');

      const tops = await page.evaluate(() =>
        [...document.querySelectorAll('.toc-rail__tick')].map((t) => parseFloat(t.style.top)));
      assert.ok(tops.every((t, i) => i === 0 || t >= tops[i - 1]), 'ticks in document order');
      assert.ok(Math.max(...tops) <= 100 && Math.min(...tops) >= 0);
    });
  } finally {
    server.close();
  }
});

test('scrolling to the bottom activates the last heading', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('spy-end', headingSeries(20), { filler: 60 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      const ids = await page.evaluate(() =>
        window.TocPanel.instance.entries.map((e) => e.id));
      await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
      await page.waitForTimeout(300);
      assert.equal(await activeId(page), ids[ids.length - 1]);
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — no element matches `.toc-panel__item[data-active="true"]`.

- [ ] **Step 3: Implement the scroll spy**

Add before `init`:

```js
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
```

Note `topLevelAncestorId` walks to the **outermost** `li` with a `data-toc-id`, which is the level-1 entry. Task 5 reuses it.

- [ ] **Step 4: Wire the spy into `init`**

Inside `init`, after `window.TocPanel.instance = instance;` and before `return instance;`:

```js
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

        doc.dispatchEvent(new CustomEvent('toc:active', { detail: { id: id, topId: topId } }));
      },
      rails.position
    );
```

`rails.position` is passed as the fourth argument, `onMeasure`, **not** called from inside the `onChange` callback above. Ticks' `top%` depends only on document height and heading offsets — both invariant under pure scrolling — so recomputing them on every scroll-driven active-heading change is redundant work today and goes stale the day a MathJax reflow moves headings with no scroll at all. Get this backwards and the "still responsive" check on the largest real page (Task 11, several hundred headings) is the symptom that catches it.

- [ ] **Step 5: Append the tick CSS**

```css
.toc-rail__tick {
  position: absolute;
  right: 7px;
  width: 6px;
  height: 2px;
  margin-top: -1px;
  border-radius: 1px;
  background: var(--line);
  transition: width 0.18s ease, background 0.18s ease;
}

.toc-rail__tick--active {
  width: 13px;
  background: var(--accent);
}

.toc-panel__item[data-active="true"] > .toc-panel__link {
  color: var(--accent);
  border-left-color: var(--accent);
  background: var(--accent-muted);
}
```

- [ ] **Step 6: Run to verify it passes**

Run: `npm test`
Expected: PASS, 19 tests once the `onMeasure` threading and the `atBottom` scroll-range guard above are in place (17 on the first pass, before those two review fixes).

- [ ] **Step 7: Commit**

```bash
git add assets/js/toc-panel.js assets/css/site.css
git commit -m "feat: track the active heading and draw rail ticks"
```

---

### Task 5: Accordion behaviour

**Files:**
- Modify: `assets/js/toc-panel.js`
- Test: `tests/toc-accordion.spec.mjs`

**Interfaces:**
- Consumes: `toc:active` event and `topLevelAncestorId` from Task 4; `.toc-panel__twisty` from Task 3.
- Produces: `createAccordion(listEl) => {syncTo(id, topId), expandAll(), restore(), setExpanded(li, expanded)}` on `instance.accordion`. `li[data-user-locked="true"]` marks a manually toggled branch. `setExpanded` is returned alongside the other three even though nothing outside this module calls it directly today.

- [ ] **Step 1: Write the failing test**

Create `tests/toc-accordion.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture, headingSeries } from './helpers.mjs';

const visibleLinks = (page) =>
  page.evaluate(
    () => [...document.querySelectorAll('.toc-panel__link')]
      .filter((a) => a.getBoundingClientRect().height > 0).length
  );

test('a 720-heading page collapses to a readable number of rows', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('acc-big', headingSeries(720, 12), { filler: 3 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      // Panel must be open for rows to have layout height.
      await page.evaluate(() => document.documentElement.setAttribute('data-toc-mode', 'pinned'));

      const shown = await visibleLinks(page);
      assert.ok(shown < 90, `expected far fewer than 720 rows, saw ${shown}`);
      assert.ok(shown >= 60, `all 60 chapters should still be listed, saw ${shown}`);
    });
  } finally {
    server.close();
  }
});

test('the branch containing the active heading expands, others collapse', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('acc-sync', headingSeries(36, 12), { filler: 60 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      await page.evaluate(() => document.documentElement.setAttribute('data-toc-mode', 'pinned'));

      const ids = await page.evaluate(() =>
        window.TocPanel.instance.entries.map((e) => ({ id: e.id, level: e.level })));
      const secondChapter = ids.filter((e) => e.level === 1)[1].id;
      const inSecond = ids[ids.findIndex((e) => e.id === secondChapter) + 2].id;

      await page.evaluate((id) => {
        const el = document.getElementById(id);
        window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 20);
      }, inSecond);
      await page.waitForTimeout(300);

      const state = await page.evaluate((chapterId) => {
        const li = (id) => document.querySelector(`li[data-toc-id="${id}"]`);
        const chapters = [...document.querySelectorAll('.toc-panel__list > li')];
        return {
          activeExpanded: li(chapterId).getAttribute('data-expanded'),
          othersExpanded: chapters
            .filter((c) => c.getAttribute('data-toc-id') !== chapterId)
            .map((c) => c.getAttribute('data-expanded'))
        };
      }, secondChapter);

      assert.equal(state.activeExpanded, 'true');
      assert.ok(state.othersExpanded.every((v) => v !== 'true'), 'other chapters collapsed');
    });
  } finally {
    server.close();
  }
});

test('clicking a twisty overrides auto-collapse until the chapter changes', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('acc-manual', headingSeries(36, 12), { filler: 60 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      await page.evaluate(() => document.documentElement.setAttribute('data-toc-mode', 'pinned'));

      const ids = await page.evaluate(() =>
        window.TocPanel.instance.entries.filter((e) => e.level === 1).map((e) => e.id));
      const third = ids[2];

      await page.click(`li[data-toc-id="${third}"] > .toc-panel__twisty`);
      let state = await page.evaluate((id) => ({
        expanded: document.querySelector(`li[data-toc-id="${id}"]`).getAttribute('data-expanded'),
        locked: document.querySelector(`li[data-toc-id="${id}"]`).getAttribute('data-user-locked'),
        aria: document
          .querySelector(`li[data-toc-id="${id}"] > .toc-panel__twisty`)
          .getAttribute('aria-expanded')
      }), third);

      assert.equal(state.expanded, 'true');
      assert.equal(state.locked, 'true');
      assert.equal(state.aria, 'true');

      // Scroll into a different chapter: the manual lock is released.
      const otherId = ids[1];
      await page.evaluate((id) => {
        const el = document.getElementById(id);
        window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY + 10);
      }, otherId);
      await page.waitForTimeout(300);

      state = await page.evaluate((id) => ({
        expanded: document.querySelector(`li[data-toc-id="${id}"]`).getAttribute('data-expanded'),
        locked: document.querySelector(`li[data-toc-id="${id}"]`).getAttribute('data-user-locked')
      }), third);

      assert.notEqual(state.expanded, 'true');
      assert.notEqual(state.locked, 'true');
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — every chapter still reports `data-expanded="false"` after scrolling.

- [ ] **Step 3: Implement the accordion**

Add before `init`:

```js
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
      var twisty = event.target.closest('.toc-panel__twisty');
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
```

- [ ] **Step 4: Hook it to the active-heading callback**

In `init`, create the accordion before the spy:

```js
    instance.accordion = createAccordion(instance.listEl);
```

and inside the spy's `onChange`, after the tick update and before the `dispatchEvent(new CustomEvent('toc:active', ...))` call:

```js
        instance.accordion.syncTo(id, topId);
```

(Task 4's `onChange` no longer calls `rails.position()` itself — that moved to `measure()` via the `onMeasure` callback — so the tick update is directly followed by this line and then the dispatch.)

- [ ] **Step 5: Run to verify it passes**

Run: `npm test`
Expected: PASS, 23 tests. (22 on the first pass; the review added one more test locking a level-2 branch and then scrolling within the same chapter, so `clearLocks` genuinely does not fire and `syncTo` is exercised with the locked branch outside the active ancestor chain.)

- [ ] **Step 6: Commit**

```bash
git add assets/js/toc-panel.js
git commit -m "feat: collapse the TOC tree to the section being read"
```

---

### Task 6: Reveal controller — hover, hot zone, pinning

**Files:**
- Modify: `assets/js/toc-panel.js`, `assets/css/site.css`
- Test: `tests/toc-reveal.spec.mjs`

**Interfaces:**
- Consumes: `.toc-root`, `.toc-root__hotzone`, `.toc-rail`, `.toc-panel` from Task 3.
- Produces: `createReveal(instance) => {open(), close(), toggle(), mode()}` on `instance.reveal`.
  - `documentElement` carries `data-toc-mode` ∈ `pinned | overlay | mobile` and `data-toc-open` ∈ `true | false`.
  - Close delay constant `CLOSE_DELAY = 250`.

- [ ] **Step 1: Write the failing test**

Create `tests/toc-reveal.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture, headingSeries } from './helpers.mjs';

const mode = (page) => page.evaluate(() => document.documentElement.getAttribute('data-toc-mode'));
const isOpen = (page) => page.evaluate(() => document.documentElement.getAttribute('data-toc-open') === 'true');
const panelX = (page) => page.evaluate(() => document.querySelector('.toc-panel').getBoundingClientRect().left);

test('mode is width-driven: pinned at 1660, overlay between, mobile below 900', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('reveal-mode', headingSeries(24), { filler: 20 });

    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await mode(page), 'pinned');
    });
    await withPage({ origin: server.origin, viewport: { width: 1280, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await mode(page), 'overlay');
      assert.equal(await isOpen(page), false);
    });
    await withPage({ origin: server.origin, viewport: { width: 420, height: 800 } }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await mode(page), 'mobile');
    });
  } finally {
    server.close();
  }
});

test('pinned mode leaves the panel on screen and clear of the text column', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('reveal-pinned', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await panelX(page), 1700 - 280);

      const textRight = await page.evaluate(
        () => document.querySelector('.page-content .wrapper').getBoundingClientRect().right
      );
      assert.ok(textRight <= 1700 - 280, `text ends at ${textRight}, panel starts at ${1700 - 280}`);
    });
  } finally {
    server.close();
  }
});

test('hovering the right edge opens the panel, leaving closes it', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('reveal-hover', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: { width: 1280, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await isOpen(page), false);

      await page.mouse.move(1274, 450);
      await page.waitForFunction(
        () => document.documentElement.getAttribute('data-toc-open') === 'true',
        null, { timeout: 2000 }
      );
      assert.equal(await panelX(page), 1280 - 280);

      await page.mouse.move(300, 450);
      await page.waitForFunction(
        () => document.documentElement.getAttribute('data-toc-open') !== 'true',
        null, { timeout: 2000 }
      );
    });
  } finally {
    server.close();
  }
});

test('clicking a heading closes the overlay but not the pinned panel', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('reveal-click', headingSeries(24), { filler: 40 });

    await withPage({ origin: server.origin, viewport: { width: 1280, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      await page.mouse.move(1274, 450);
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') === 'true');
      await page.click('.toc-panel__list > li > .toc-panel__link');
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') !== 'true');
    });

    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      await page.click('.toc-panel__list > li > .toc-panel__link');
      await page.waitForTimeout(400);
      assert.equal(await mode(page), 'pinned');
      assert.equal(await panelX(page), 1700 - 280);
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `data-toc-mode` is `null`.

- [ ] **Step 3: Implement the reveal controller**

Add before `init`:

```js
  var CLOSE_DELAY = 250;
  // 1660 = .wrapper's 1100px content column + a 280px panel on each side, so a
  // pinned panel never overlaps the card. At 1600 the panel would cover the
  // rightmost 30px of the column and clip anything overflowing it (wide
  // tables, long code lines). Keep this in step with --toc-panel-width and
  // .wrapper's max-width in site.css.
  var PIN_QUERY = '(min-width: 1660px)';
  var MOBILE_QUERY = '(max-width: 899px)';

  function createReveal(instance, doc) {
    var rootEl = doc.documentElement;
    var pinMedia = window.matchMedia(PIN_QUERY);
    var mobileMedia = window.matchMedia(MOBILE_QUERY);
    var timer = 0;

    function mode() {
      if (mobileMedia.matches) return 'mobile';
      if (pinMedia.matches) return 'pinned';
      return 'overlay';
    }

    function applyMode() {
      var next = mode();
      rootEl.setAttribute('data-toc-mode', next);
      if (next === 'pinned') {
        rootEl.setAttribute('data-toc-open', 'true');
      } else if (rootEl.getAttribute('data-toc-open') !== 'true') {
        rootEl.setAttribute('data-toc-open', 'false');
      }
    }

    function cancelClose() {
      if (timer) { window.clearTimeout(timer); timer = 0; }
    }

    function open() {
      cancelClose();
      rootEl.setAttribute('data-toc-open', 'true');
    }

    function close(immediate) {
      cancelClose();
      if (mode() === 'pinned') return;
      if (immediate) {
        rootEl.setAttribute('data-toc-open', 'false');
      } else {
        timer = window.setTimeout(function () {
          timer = 0;
          if (mode() !== 'pinned') rootEl.setAttribute('data-toc-open', 'false');
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
      if (mode() !== 'pinned') close(true);
    });

    applyMode();
    if (pinMedia.addEventListener) {
      pinMedia.addEventListener('change', applyMode);
      mobileMedia.addEventListener('change', applyMode);
    } else if (pinMedia.addListener) {
      pinMedia.addListener(applyMode);
      mobileMedia.addListener(applyMode);
    }

    return { open: open, close: close, toggle: toggle, mode: mode };
  }
```

- [ ] **Step 4: Wire it into `init`**

Do **not** add this after the spy. Wire it immediately after `window.TocPanel.instance = instance;` — before `renderTicks`, `createAccordion`, and `createScrollSpy` are called:

```js
    instance.reveal = createReveal(instance, doc);
```

This ordering is load-bearing, not stylistic. `renderTicks` and `createScrollSpy` both read layout (`getBoundingClientRect()`), which forces the browser to commit `.toc-panel`'s off-screen `transform: translateX(100%)` as an already-rendered style. If that layout read happened first, `createReveal`'s pinned-mode attribute flip would be a *change* from a real prior frame, and the panel's 0.22s CSS transition (Task 3) would genuinely animate it into view on every page load. Wiring reveal first means the very first style computation for the freshly-inserted `.toc-panel` already bakes in the resting position, so there is nothing to transition from and no on-load flash. See Deviation 4 in "Deviations from the spec" above.

- [ ] **Step 5: Append the reveal CSS**

```css
:root[data-toc-open="true"] .toc-panel,
:root[data-toc-mode="pinned"] .toc-panel {
  transform: none;
}

:root[data-toc-mode="pinned"] .toc-root__hotzone {
  display: none;
}

:root[data-toc-open="true"] .toc-rail__tick {
  opacity: 0;
}
```

- [ ] **Step 6: Run to verify it passes**

Run: `npm test`
Expected: PASS, 28 tests. (27 on the first pass; no test up to that point could tell an eased slide apart from an instant snap, so the review round added one that can before checking anything else.)

**Watch for this while implementing:** it is tempting to suppress `.toc-panel`'s transition before flipping `data-toc-open` and force a synchronous reflow, so the open state is guaranteed correct the instant a test asserts on it. Do not do this in `open()` — it makes every hover-triggered reveal snap open instantly with nothing left to animate, which destroys the slide the human partner chose. That exact shortcut shipped once and passed every existing test green precisely because none of them could distinguish a slide from a jump. If a test's timing assertion needs help, fix the test (arm a `transitionend` listener before the triggering event, with a bounded timeout), not the production code.

- [ ] **Step 7: Commit**

```bash
git add assets/js/toc-panel.js assets/css/site.css
git commit -m "feat: reveal the panel on right-edge hover, pin it above 1660px"
```

---

### Task 7: Mobile drawer

**Files:**
- Modify: `assets/js/toc-panel.js`, `assets/css/site.css`
- Test: `tests/toc-mobile.spec.mjs`

**Interfaces:**
- Consumes: `createReveal` from Task 6.
- Produces: DOM `button.toc-fab` and `div.toc-backdrop`, both children of `.toc-root`; created in `renderShell`, exposed as `instance.fab` and `instance.backdrop`.

- [ ] **Step 1: Write the failing test**

Create `tests/toc-mobile.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture, headingSeries } from './helpers.mjs';

const PHONE = { width: 420, height: 800 };

test('the rail is hidden and a button is offered on phones', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('mobile-fab', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: PHONE }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await page.locator('.toc-rail').isVisible(), false);
      assert.equal(await page.locator('.toc-fab').isVisible(), true);
      assert.equal(await page.locator('.toc-backdrop').isVisible(), false);

      const box = await page.locator('.toc-fab').boundingBox();
      assert.ok(box.x + box.width > PHONE.width - 90, 'button hugs the right edge');
      assert.ok(box.y + box.height > PHONE.height - 120, 'button sits near the bottom');
    });
  } finally {
    server.close();
  }
});

test('tapping the button opens a drawer that a backdrop tap closes', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('mobile-drawer', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: PHONE }, async (page) => {
      await page.goto(server.origin + url);

      await page.click('.toc-fab');
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') === 'true');
      assert.equal(await page.locator('.toc-backdrop').isVisible(), true);
      assert.equal(await page.locator('.toc-fab').getAttribute('aria-expanded'), 'true');

      await page.locator('.toc-backdrop').click({ position: { x: 20, y: 200 } });
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') !== 'true');
      assert.equal(await page.locator('.toc-fab').getAttribute('aria-expanded'), 'false');
    });
  } finally {
    server.close();
  }
});

test('tapping a heading in the drawer closes it', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('mobile-jump', headingSeries(24), { filler: 40 });
    await withPage({ origin: server.origin, viewport: PHONE }, async (page) => {
      await page.goto(server.origin + url);
      await page.click('.toc-fab');
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') === 'true');
      await page.click('.toc-panel__list > li > .toc-panel__link');
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') !== 'true');
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `.toc-fab` not found.

- [ ] **Step 3: Add the button and backdrop to `renderShell`**

Inside `renderShell`, before the `return`:

```js
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
```

and extend the returned object with `backdrop: backdrop, fab: fab`. In `init`, copy both onto `instance`:

```js
      backdrop: shell.backdrop,
      fab: shell.fab,
```

- [ ] **Step 4: Extend `createReveal` for the drawer**

Inside `createReveal`, after the existing listeners:

```js
    instance.fab.addEventListener('click', function () { toggle(); });
    instance.backdrop.addEventListener('click', function () { close(true); });
```

and make `open` / `close` keep the button's ARIA state in sync by replacing the two `setAttribute('data-toc-open', …)` bodies with a shared setter:

```js
    function setOpen(next) {
      rootEl.setAttribute('data-toc-open', next ? 'true' : 'false');
      instance.fab.setAttribute('aria-expanded', next ? 'true' : 'false');
      instance.fab.setAttribute(
        'aria-label',
        next ? 'Close table of contents' : 'Open table of contents'
      );
    }
```

Use `setOpen(true)` in `open()` and `applyMode()`'s pinned branch, and `setOpen(false)` everywhere the panel closes.

- [ ] **Step 5: Append the mobile CSS**

```css
.toc-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.45);
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.2s ease, visibility 0.2s ease;
  pointer-events: none;
}

.toc-fab {
  display: none;
  position: fixed;
  right: 1rem;
  bottom: 1rem;
  width: 46px;
  height: 46px;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow);
  pointer-events: auto;
  cursor: pointer;
  z-index: 1;
}

:root[data-toc-mode="mobile"] .toc-rail,
:root[data-toc-mode="mobile"] .toc-root__hotzone {
  display: none;
}

:root[data-toc-mode="mobile"] .toc-fab {
  display: flex;
}

:root[data-toc-mode="mobile"] .toc-root {
  width: min(320px, 85vw);
}

:root[data-toc-mode="mobile"][data-toc-open="true"] .toc-backdrop {
  opacity: 1;
  visibility: visible;
  pointer-events: auto;
}

:root[data-toc-mode="mobile"][data-toc-open="true"] .toc-fab {
  display: none;
}
```

`.toc-backdrop` is `position: fixed; inset: 0` inside a `pointer-events: none` parent, so it only receives taps once the rule above turns `pointer-events` back on.

- [ ] **Step 6: Run to verify it passes**

Run: `npm test`
Expected: PASS, 31 tests.

- [ ] **Step 7: Commit**

```bash
git add assets/js/toc-panel.js assets/css/site.css
git commit -m "feat: offer the TOC as a drawer on narrow viewports"
```

---

### Task 8: Filter

**Files:**
- Modify: `assets/js/toc-panel.js`, `assets/css/site.css`
- Test: `tests/toc-filter.spec.mjs`

**Interfaces:**
- Consumes: `filterKey` from Task 2, `instance.accordion` from Task 5, `instance.filterEl` from Task 3.
- Produces: `createFilter(instance) => {apply(query), clear(), matches()}` on `instance.filter`. `matches()` returns the visible `a.toc-panel__link` elements in document order — Task 9 consumes this for arrow-key navigation.
- DOM: `li[data-filtered="out"]` is hidden; `:root[data-toc-filtering="true"]` while a query is active.

- [ ] **Step 1: Write the failing test**

Create `tests/toc-filter.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture } from './helpers.mjs';

const HEADINGS = [
  { level: 1, text: 'Chapter 2: Sparse Recovery' },
  { level: 2, text: '2.2.2 The $\\ell^0$ Norm' },
  { level: 2, text: '2.2.3 The Sparsest Solution' },
  { level: 3, text: 'Computational Complexity' },
  { level: 1, text: 'Chapter 3: Convex Relaxation' },
  { level: 2, text: 'Solution for $V^\\pi$' },
  { level: 2, text: 'Basis Pursuit' }
];

const visibleTexts = (page) =>
  page.evaluate(() =>
    [...document.querySelectorAll('.toc-panel__link')]
      .filter((a) => a.getBoundingClientRect().height > 0)
      .map((a) => a.textContent.trim())
  );

test('typing narrows the tree and keeps matching ancestors', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('filter-basic', HEADINGS, { filler: 10 });
    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      await page.fill('.toc-panel__filter', 'sparsest');

      const shown = await visibleTexts(page);
      assert.ok(shown.some((t) => t.includes('Sparsest')), 'the match is visible');
      assert.ok(shown.some((t) => t.includes('Chapter 2')), 'its parent stays visible for context');
      assert.ok(!shown.some((t) => t.includes('Basis Pursuit')), 'non-matches are hidden');
    });
  } finally {
    server.close();
  }
});

test('math headings are findable by their plain text', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('filter-math', HEADINGS, { filler: 10 });
    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);

      await page.fill('.toc-panel__filter', 'norm');
      let shown = await visibleTexts(page);
      assert.ok(shown.some((t) => t.includes('Norm')), '"norm" should find the $\\ell^0$ heading');

      await page.fill('.toc-panel__filter', 'solution for');
      shown = await visibleTexts(page);
      assert.ok(shown.some((t) => t.includes('Solution for')), 'text around math still matches');

      await page.fill('.toc-panel__filter', '2.2.2');
      shown = await visibleTexts(page);
      assert.ok(shown.some((t) => t.includes('2.2.2')), 'numbering matches');
    });
  } finally {
    server.close();
  }
});

test('clearing the filter restores the accordion', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('filter-clear', HEADINGS, { filler: 10 });
    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      const before = await visibleTexts(page);

      await page.fill('.toc-panel__filter', 'basis');
      assert.notDeepEqual(await visibleTexts(page), before);

      await page.fill('.toc-panel__filter', '');
      await page.waitForTimeout(150);
      assert.deepEqual(await visibleTexts(page), before);
      assert.equal(
        await page.evaluate(() => document.documentElement.getAttribute('data-toc-filtering')),
        'false'
      );
    });
  } finally {
    server.close();
  }
});

test('a query with no matches shows nothing rather than everything', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('filter-empty', HEADINGS, { filler: 10 });
    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      await page.fill('.toc-panel__filter', 'zzzzz');
      assert.deepEqual(await visibleTexts(page), []);
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — typing changes nothing.

- [ ] **Step 3: Implement the filter**

Add before `init`:

```js
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

    rootEl.setAttribute('data-toc-filtering', 'false');

    return {
      apply: apply,
      clear: function () { instance.filterEl.value = ''; clear(); },
      matches: matches,
      get active() { return active; }
    };
  }
```

Note `clear()` re-syncs the accordion directly, using `instance.spy.activeId` and `topLevelAncestorId` (Task 4), rather than calling `instance.spy.refresh()`. `refresh()` looks like the obvious fix — it re-runs `measure()` and `pick()` — but `pick()` only calls `onChange` (which is what calls `accordion.syncTo`) when the active heading id **changes**, and clearing a filter does not move the reader, so the id is unchanged and `onChange` never fires. Combined with `accordion.restore()` not collapsing anything on its own, that combination would leave every branch expanded from the filter's `expandAll()` forever. This is what the "restores the accordion" test asserts, and it fails against the `restore()` + `refresh()` form for exactly this reason.

- [ ] **Step 4: Wire it into `init`**

After `instance.reveal = createReveal(instance, doc);`:

```js
    instance.filter = createFilter(instance, doc);
```

- [ ] **Step 5: Append the filter CSS**

```css
.toc-panel__item[data-filtered="out"] {
  display: none;
}
```

- [ ] **Step 6: Run to verify it passes**

Run: `npm test`
Expected: PASS, 35 tests.

- [ ] **Step 7: Commit**

```bash
git add assets/js/toc-panel.js assets/css/site.css
git commit -m "feat: filter the TOC, matching math headings by plain text"
```

---

### Task 9: Keyboard access

**Files:**
- Modify: `assets/js/toc-panel.js`, `assets/css/site.css`
- Test: `tests/toc-keyboard.spec.mjs`

**Interfaces:**
- Consumes: `instance.reveal`, `instance.filter.matches()`, `instance.filterEl`.
- Produces: `createKeyboard(instance, doc)` on `instance.keyboard`. Highlighted row carries `data-kbd="true"`.

- [ ] **Step 1: Write the failing test**

Create `tests/toc-keyboard.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture, headingSeries } from './helpers.mjs';

const DESKTOP = { width: 1280, height: 900 };

test('slash and Cmd+K open the panel with the filter focused', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('kbd-open', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: DESKTOP }, async (page) => {
      await page.goto(server.origin + url);

      await page.keyboard.press('/');
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') === 'true');
      assert.equal(await page.evaluate(() => document.activeElement.className), 'toc-panel__filter');
      assert.equal(await page.inputValue('.toc-panel__filter'), '', 'the slash must not be typed');

      await page.keyboard.press('Escape');
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') !== 'true');

      await page.keyboard.press('ControlOrMeta+k');
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') === 'true');
    });
  } finally {
    server.close();
  }
});

test('slash typed inside an input is left alone', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('kbd-input', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: DESKTOP }, async (page) => {
      await page.goto(server.origin + url);
      await page.evaluate(() => {
        const input = document.createElement('input');
        input.id = 'probe';
        document.querySelector('.page-content .wrapper').prepend(input);
        input.focus();
      });

      await page.keyboard.press('/');
      assert.equal(await page.inputValue('#probe'), '/');
      assert.equal(
        await page.evaluate(() => document.documentElement.getAttribute('data-toc-open')),
        'false'
      );
    });
  } finally {
    server.close();
  }
});

test('arrows walk the filtered results and Enter jumps', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('kbd-arrows', headingSeries(36, 12), { filler: 40 });
    await withPage({ origin: server.origin, viewport: DESKTOP }, async (page) => {
      await page.goto(server.origin + url);
      await page.keyboard.press('/');
      await page.waitForFunction(() => document.documentElement.getAttribute('data-toc-open') === 'true');

      await page.keyboard.type('topic 1');
      await page.waitForTimeout(120);

      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('ArrowDown');
      const highlighted = await page.evaluate(() => {
        const li = document.querySelector('li[data-kbd="true"]');
        return li ? li.getAttribute('data-toc-id') : null;
      });
      assert.ok(highlighted, 'a row should be highlighted');

      await page.keyboard.press('Enter');
      await page.waitForTimeout(400);
      assert.equal(await page.evaluate(() => location.hash.slice(1)), highlighted);
    });
  } finally {
    server.close();
  }
});

test('the panel exposes navigation semantics', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('kbd-a11y', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: DESKTOP }, async (page) => {
      await page.goto(server.origin + url);
      assert.equal(await page.getAttribute('.toc-panel', 'role'), 'navigation');
      assert.equal(await page.getAttribute('.toc-panel', 'aria-label'), 'Table of contents');
      assert.equal(await page.getAttribute('.toc-panel__filter', 'aria-label'), 'Filter sections');
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — pressing `/` does nothing.

- [ ] **Step 3: Implement keyboard handling**

Add before `init`:

```js
  function isTypingTarget(el) {
    if (!el) return false;
    var tag = el.tagName;
    return (
      tag === 'INPUT' ||
      tag === 'TEXTAREA' ||
      tag === 'SELECT' ||
      el.isContentEditable === true
    );
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

    function highlight(index) {
      var list = rows();
      var previous = instance.listEl.querySelector('li[data-kbd="true"]');
      if (previous) previous.removeAttribute('data-kbd');

      if (!list.length) { cursor = -1; return; }

      cursor = Math.max(0, Math.min(index, list.length - 1));
      var li = list[cursor].parentElement;
      li.setAttribute('data-kbd', 'true');
      if (li.scrollIntoView) li.scrollIntoView({ block: 'nearest' });
    }

    function openForSearch() {
      invalidateRows();
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

      if ((event.metaKey || event.ctrlKey) && (event.key === 'k' || event.key === 'K')) {
        event.preventDefault();
        openForSearch();
        return;
      }

      if (event.key === 'Escape') {
        if (instance.filter.active) instance.filter.clear();
        instance.reveal.close(true);
        invalidateRows();
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
```

- [ ] **Step 4: Wire it into `init`**

After `instance.filter = createFilter(instance, doc);`:

```js
    instance.keyboard = createKeyboard(instance, doc);
```

- [ ] **Step 5: Append the highlight CSS**

```css
.toc-panel__item[data-kbd="true"] > .toc-panel__link {
  background: var(--accent-muted);
  color: var(--accent);
  outline: 1px solid var(--accent);
}
```

- [ ] **Step 6: Run to verify it passes**

Run: `npm test`
Expected: PASS, 43 tests. (40 on the first pass; the review found a Critical — the row cache above was invalidated only at this module's own touchpoints, so a scroll-driven `syncTo` collapsing a branch, or a manual twisty click, left arrow-key navigation walking a stale row list. The `MutationObserver` above is the fix, and it earned three new tests.)

- [ ] **Step 7: Commit**

```bash
git add assets/js/toc-panel.js assets/css/site.css
git commit -m "feat: open, filter and jump the TOC from the keyboard"
```

---

### Task 10: Heading anchors

**Files:**
- Modify: `assets/js/heading-anchors.js`, `assets/css/site.css`
- Test: `tests/heading-anchors.spec.mjs`

**Interfaces:**
- Consumes: nothing from the panel. Independent by design.
- Produces: `a.heading-anchor[href="#id"]` appended to every `h1`–`h3` with an id inside `.page-content`. Copied state is `data-copied="true"` for 1200 ms.

- [ ] **Step 1: Write the failing test**

Create `tests/heading-anchors.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture, headingSeries } from './helpers.mjs';

test('every content heading with an id gets an anchor, and the panel does not', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('anchors', headingSeries(24), { filler: 10 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);

      const headings = await page.locator('.page-content :is(h1,h2,h3)[id]').count();
      assert.equal(await page.locator('.page-content .heading-anchor').count(), headings);
      assert.equal(await page.locator('.toc-panel .heading-anchor').count(), 0);

      const href = await page.getAttribute('.page-content h1 .heading-anchor', 'href');
      const id = await page.getAttribute('.page-content h1', 'id');
      assert.equal(href, '#' + id);
    });
  } finally {
    server.close();
  }
});

test('clicking an anchor copies the absolute link', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('anchors-copy', headingSeries(24), { filler: 10 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
      await page.goto(server.origin + url);

      const id = await page.getAttribute('.page-content h1', 'id');
      await page.click('.page-content h1 .heading-anchor');

      const copied = await page.evaluate(() => navigator.clipboard.readText());
      assert.equal(copied, server.origin + url + '#' + id);

      assert.equal(
        await page.getAttribute('.page-content h1 .heading-anchor', 'data-copied'),
        'true'
      );
      await page.waitForTimeout(1400);
      assert.equal(
        await page.getAttribute('.page-content h1 .heading-anchor', 'data-copied'),
        null
      );
    });
  } finally {
    server.close();
  }
});

test('anchors are hidden until their heading is hovered', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('anchors-hover', headingSeries(24), { filler: 10 });
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + url);
      const opacity = () =>
        page.evaluate(() =>
          getComputedStyle(document.querySelector('.page-content h1 .heading-anchor')).opacity);

      assert.equal(await opacity(), '0');
      await page.hover('.page-content h1');
      await page.waitForTimeout(250);
      assert.notEqual(await opacity(), '0');
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — `.heading-anchor` count is 0.

- [ ] **Step 3: Implement `assets/js/heading-anchors.js`**

```js
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
    anchor.textContent = '\u00B6';
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
```

`writeText(...).then(...)` is called with both the fulfilled and the rejected handler, not just the first. A permissions-denied clipboard write is a real browser outcome, not a hypothetical: `preventDefault()` above already cancelled the anchor's own navigation, so without a rejection handler a denied write is a silent dead end — no feedback, and an unhandled promise rejection to boot. The rejection handler falls back to an ordinary same-page navigation to the fragment, so the click still does something. `clearTimer` is scoped inside `decorate()`, so it is per-anchor: rapid repeated clicks on the *same* pilcrow clear and restart their own timer, without touching any other heading's `data-copied` state.

Because `toc-panel.js` moves `#markdown-toc` out of `.page-content` on `DOMContentLoaded`, and its script tag runs first, the TOC links are already gone from `.page-content` by the time this runs — so the panel never receives `¶` marks.

- [ ] **Step 4: Append the anchor CSS**

```css
.heading-anchor {
  margin-left: 0.4em;
  color: var(--muted);
  font-weight: 400;
  font-size: 0.75em;
  text-decoration: none;
  opacity: 0;
  transition: opacity 0.15s ease, color 0.15s ease;
}

.post-content h1:hover .heading-anchor,
.post-content h2:hover .heading-anchor,
.post-content h3:hover .heading-anchor,
.page-content h1:hover .heading-anchor,
.page-content h2:hover .heading-anchor,
.page-content h3:hover .heading-anchor,
.heading-anchor:focus {
  opacity: 1;
}

.heading-anchor:hover {
  color: var(--accent);
  text-decoration: none;
}

.heading-anchor[data-copied="true"] {
  opacity: 1;
  color: var(--accent);
}

.heading-anchor[data-copied="true"]::after {
  content: " copied";
  font-size: 0.85em;
  letter-spacing: 0.04em;
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `npm test`
Expected: PASS, 47 tests. (46 on the first pass; the review's fix round added a rejection-path test for the clipboard write.)

- [ ] **Step 6: Commit**

```bash
git add assets/js/heading-anchors.js assets/css/site.css
git commit -m "feat: add copyable heading anchors"
```

---

### Task 11: Theme, print, reduced motion, and real-page verification

**Files:**
- Modify: `assets/css/site.css`
- Test: `tests/toc-presentation.spec.mjs`, `tests/real-pages.spec.mjs`

**Interfaces:**
- Consumes: everything above.
- Produces: no new JS API. Final CSS media rules.

- [ ] **Step 1: Write the failing presentation test**

Create `tests/toc-presentation.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startServer, withPage, buildFixture, headingSeries } from './helpers.mjs';

test('the panel follows the dark theme', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('theme', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      const light = await page.evaluate(
        () => getComputedStyle(document.querySelector('.toc-panel')).backgroundColor);

      await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
      const dark = await page.evaluate(
        () => getComputedStyle(document.querySelector('.toc-panel')).backgroundColor);

      assert.notEqual(light, dark, 'panel background must change with the theme');
      assert.equal(dark, 'rgb(15, 23, 42)', 'should resolve to --surface in dark mode');
    });
  } finally {
    server.close();
  }
});

test('print flattens the panel and drops the chrome', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('print', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: { width: 1280, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      await page.emulateMedia({ media: 'print' });

      const style = await page.evaluate(() => {
        const panel = getComputedStyle(document.querySelector('.toc-panel'));
        return {
          position: panel.position,
          transform: panel.transform,
          rail: getComputedStyle(document.querySelector('.toc-rail')).display,
          fab: getComputedStyle(document.querySelector('.toc-fab')).display,
          filter: getComputedStyle(document.querySelector('.toc-panel__filter')).display,
          anchor: getComputedStyle(document.querySelector('.heading-anchor')).display
        };
      });

      assert.equal(style.position, 'static', 'panel must join the flow when printed');
      assert.ok(style.transform === 'none' || style.transform === 'matrix(1, 0, 0, 1, 0, 0)');
      assert.equal(style.rail, 'none');
      assert.equal(style.fab, 'none');
      assert.equal(style.filter, 'none');
      assert.equal(style.anchor, 'none');

      const collapsed = await page.evaluate(
        () => [...document.querySelectorAll('.toc-panel__sublist')]
          .some((ul) => getComputedStyle(ul).display === 'none'));
      assert.equal(collapsed, false, 'every branch must be expanded when printed');
    });
  } finally {
    server.close();
  }
});

test('reduced motion removes the slide', async () => {
  const server = await startServer();
  try {
    const url = buildFixture('motion', headingSeries(24), { filler: 20 });
    await withPage({ origin: server.origin, viewport: { width: 1280, height: 900 } }, async (page) => {
      await page.goto(server.origin + url);
      await page.emulateMedia({ reducedMotion: 'reduce' });
      const duration = await page.evaluate(
        () => getComputedStyle(document.querySelector('.toc-panel')).transitionDuration);
      assert.equal(duration, '0s');
    });
  } finally {
    server.close();
  }
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — panel `position` is `absolute` under print media.

- [ ] **Step 3: Append the presentation CSS**

```css
@media (prefers-reduced-motion: reduce) {
  .toc-panel,
  .toc-rail__tick,
  .toc-backdrop,
  .heading-anchor,
  .toc-panel__twisty::before {
    transition: none;
  }
}

@media print {
  .toc-root {
    position: static;
    width: auto;
    pointer-events: auto;
  }

  .toc-rail,
  .toc-root__hotzone,
  .toc-fab,
  .toc-backdrop,
  .toc-panel__header,
  .toc-panel__filter,
  .toc-panel__twisty,
  .heading-anchor {
    display: none !important;
  }

  .toc-panel {
    position: static;
    width: auto;
    transform: none;
    transition: none;
    box-shadow: none;
    border: 0;
    background: transparent;
    display: block;
  }

  .toc-panel__body {
    overflow: visible;
    padding: 0;
  }

  /* Printing must show the whole tree regardless of accordion state. */
  .toc-panel__item--branch[data-expanded="false"] > .toc-panel__sublist,
  .toc-panel__item[data-filtered="out"] {
    display: block;
  }
}
```

The `.toc-root` rule must come before the `display: none` block so specificity is not fought over; `!important` on the hidden chrome keeps the earlier `:root[data-toc-mode="mobile"] .toc-fab { display: flex }` rule from winning. `.toc-panel`'s print rule also sets `transition: none` — see Deviation 5 above for why that is not a repeat of Task 6's Critical mistake.

There is no site-wide `html { scroll-behavior: smooth }` rule, and none should be added. It was tried and measured at 1618–1629ms for a single long jump on the 806-heading real page (`scrollHeight` 468,075px) — over a second and a half for what a navigation panel exists to make instant. The human partner ruled it out entirely; a jump through `scrollIntoView`/`location.hash` now lands in ~100ms. Because the rule never existed, the reduced-motion block above has nothing to override and carries no `scroll-behavior` line.

- [ ] **Step 4: Run to verify it passes**

Run: `npm test`
Expected: PASS, 50 tests.

- [ ] **Step 5: Rebuild the site and write the real-page test**

Run: `BUNDLE_GEMFILE=Gemfile.local bundle exec jekyll build`

Create `tests/real-pages.spec.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { startServer, withPage, REPO } from './helpers.mjs';

const BIG = '/_site/subpages/books/probabilistic-machine-learning/index.html';
const MID = '/_site/subpages/books/reinforcement_learning_overview/index.html';
const HOME = '/_site/index.html';

const built = existsSync(path.join(REPO, '_site', 'subpages', 'books',
  'probabilistic-machine-learning', 'index.html'));

test('the 719-heading page collapses and stays responsive', { skip: !built }, async () => {
  const server = await startServer();
  try {
    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + BIG);
      await page.waitForFunction(() => window.TocPanel && window.TocPanel.instance);

      const count = await page.evaluate(() => window.TocPanel.instance.entries.length);
      assert.ok(count > 400, `expected a large TOC, saw ${count}`);

      const shown = await page.evaluate(() =>
        [...document.querySelectorAll('.toc-panel__link')]
          .filter((a) => a.getBoundingClientRect().height > 0).length);
      assert.ok(shown < count / 4, `${shown} of ${count} rows visible — accordion not collapsing`);

      assert.equal(await page.locator('.page-content #markdown-toc').count(), 0);
    });
  } finally {
    server.close();
  }
});

test('a math-heavy page filters by plain text', { skip: !built }, async () => {
  const server = await startServer();
  try {
    await withPage({ origin: server.origin, viewport: { width: 1700, height: 900 } }, async (page) => {
      await page.goto(server.origin + MID);
      await page.waitForFunction(() => window.TocPanel && window.TocPanel.instance);
      await page.fill('.toc-panel__filter', 'bellman');
      const shown = await page.evaluate(() =>
        [...document.querySelectorAll('.toc-panel__link')]
          .filter((a) => a.getBoundingClientRect().height > 0)
          .map((a) => a.textContent.trim()));
      assert.ok(shown.length > 0, 'expected at least one Bellman heading');
    });
  } finally {
    server.close();
  }
});

test('a page without a TOC is untouched', { skip: !built }, async () => {
  const server = await startServer();
  try {
    await withPage({ origin: server.origin }, async (page) => {
      await page.goto(server.origin + HOME);
      assert.equal(await page.locator('.toc-root').count(), 0);
    });
  } finally {
    server.close();
  }
});
```

Note these navigate `/_site/...` directly, because the helper serves the repository root.

- [ ] **Step 6: Run the whole suite**

Run: `npm test`
Expected: PASS, 54 tests. (53 on the first implementation pass — see below for the review round that added the last one.)

**Two review findings surfaced here, both worth watching for on a re-run of this task:**

- A racy assertion in the presentation spec needed the same armed/bounded pattern as Tasks 6 and 7 (assert against an event, not a fixed `waitForTimeout`).
- The scroll latency this step's real-page run surfaces is real: measure it, don't widen a ceiling to make it pass. It is what led to the human ruling against smooth scrolling recorded in Step 3 above.

- [ ] **Step 7: Manual check against the live preview**

Run: `BUNDLE_GEMFILE=Gemfile.local bundle exec jekyll serve`

Open `http://127.0.0.1:4000/subpages/books/reinforcement_learning_overview/` and confirm by eye — the automated suite blocks the MathJax CDN, so this is the only place rendered math in the panel gets checked:

- Headings containing `$...$` render as typeset math in the panel, not raw TeX.
- Toggling the theme button repaints the panel.
- Toggling the Menlo font button changes the panel's font.
- The rail ticks land in sensible places after MathJax has finished reflowing.

- [ ] **Step 8: Commit**

```bash
git add assets/css/site.css
git commit -m "feat: theme, print and reduced-motion handling for the TOC panel"
```

- [ ] **Step 9: Confirm nothing dev-only was committed**

```bash
git status --short
git log --stat -1
```
Expected: `package.json`, `package-lock.json`, `node_modules/`, `tests/` appear in neither.

```bash
BUNDLE_GEMFILE=Gemfile.local bundle exec jekyll build
ls _site/node_modules _site/tests _site/package.json 2>&1
```
Expected: `No such file or directory` for all three.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Rail with per-chapter ticks, active tick highlighted | 4 |
| Hover rail or 24px edge zone opens; ~250ms close delay | 6 |
| Pin at ≥1660px, no manual toggle, no persistence | 6 |
| Accordion: top level always shown, active branch expands | 5 |
| Manual twisty overrides until the section changes | 5 |
| Filter field narrowing the tree | 8 |
| Click closes in overlay/drawer, stays open when pinned | 6, 7 |
| Reparent (not clone) `#markdown-toc`; hide the caption | 3 |
| No-JS fallback | 3 |
| Mobile drawer + floating button below 900px | 7 |
| `/`, `Cmd/Ctrl+K`, arrows, Enter, Esc; `/search/` unaffected | 9 |
| `role`, `aria-label`, `aria-expanded` | 3, 5, 7, 9 |
| `¶` copy-link anchors with graceful fallback | 10 |
| Existing custom properties only; dark mode free | 3, 11 |
| Print restores a flat expanded tree | 11 |
| Reduced motion | 11 |
| No-op with no TOC or fewer than 4 entries | 3 |
| Verification across the four named pages and all axes | 11 |

No gaps.

**Placeholder scan:** none. Every code step carries runnable code.

**Type consistency:** `collectEntries`/`buildTree`/`shouldActivate` (Task 2) are consumed under those exact names in Task 3. `cssEscape` and `topLevelAncestorId` are defined in Task 4 and reused in Task 5. `instance.accordion.expandAll`/`restore` (Task 5) are called in Task 8. `instance.filter.matches()` (Task 8) is called in Task 9. `instance.reveal.open`/`close` (Task 6) are called in Tasks 7 and 9. `setOpen` (Task 7) replaces the inline attribute writes introduced in Task 6 — Task 7 Step 4 says so explicitly.

**One caveat worth flagging during execution:** Task 6's tests are written against the Task 6 implementation, which sets `data-toc-open` directly. Task 7 Step 4 refactors those writes into `setOpen`. Re-run the full suite after Task 7, not just the mobile spec.

## What actually shipped

The 11 tasks above land the feature; a whole-branch review afterward — reading the finished code as a whole rather than task-by-task — found four more cross-task interaction bugs and brought the suite from 54 to 67 tests. The fixes are in `assets/js/toc-panel.js`, `assets/js/heading-anchors.js`, and `assets/css/site.css`, not in a Task 12. What follows is what a future change to this feature needs to know and cannot get from re-reading the code cold.

**Two ordering constraints in `init()` are load-bearing, not stylistic:**

1. `instance.reveal = createReveal(instance, doc);` runs immediately after `window.TocPanel.instance = instance;`, before `renderTicks`, `createAccordion`, and `createScrollSpy`. Those three read layout (`getBoundingClientRect()`), which commits `.toc-panel`'s off-screen `translateX(100%)` as an already-rendered style. Reveal after them turns the pinned-mode attribute flip into a *change* from a real prior frame, and the panel visibly slides in on every page load instead of resting in place. See Deviation 4.
2. `createFilter` is wired before `createKeyboard`. Both attach an `input` listener to the same filter field; listeners run in registration order. Filter's marks rows in/out, keyboard's then invalidates its row cache and highlights row 0. Swapped, `highlight(0)` runs against the *previous* keystroke's visible set. Nothing throws — it fails silently.

**The closed panel is `visibility: hidden`, not transform-only.** `transform: translateX(100%)` alone moves the panel off screen but leaves it in the tab order and the accessibility tree — on the largest real page that is over a thousand focusable elements a keyboard user tabs into after the footer, and a screen reader reads the whole TOC there too. `visibility` is switched instantly on close, not eased: it is held at `visible` through the whole 0.22s slide-out by a matching `transition-delay`, then drops. On open the delay is cancelled so it flips to `visible` on the same tick as `data-toc-open`. An *eased* visibility would still compute as `hidden` at t=0 on open, which silently fails `filterEl.focus()` — and with it the `/` keyboard shortcut, since focusing a hidden element is a no-op.

**Scroll-spy must not re-sync the accordion while a filter is active.** `apply()` force-expands every branch so deep matches stay visible; the spy's `onChange` calling `accordion.syncTo()` unconditionally would immediately collapse everything outside the reader's current ancestor chain, taking matched rows down with it — with the query still sitting in the filter box. This fires from ordinary scrolling in pinned mode, and even with no scrolling at all when MathJax reflow re-measures. The guard is a single `if (!(instance.filter && instance.filter.active))` around the `syncTo` call; `filter.clear()` re-syncs explicitly against the reader's real position once the filter drops (see Task 8's `clear()`).

**The test suite is deliberately gitignored.** `package.json`, `node_modules/`, and `tests/` never reach `_site` or the repository. The harness serves the **repository root**, not a Jekyll build, so a fixture under `/tests/fixtures/*.html` can reference `/assets/js/toc-panel.js` and `/assets/css/site.css` straight from the working tree — edit a script, re-run tests, no rebuild step. Only Task 11's real-page suite needs `_site` to exist, and it skips itself when that directory is absent.
