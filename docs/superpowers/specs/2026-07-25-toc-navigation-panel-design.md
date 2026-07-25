# Hover-reveal table-of-contents panel

**Date:** 2026-07-25
**Status:** approved, ready for implementation planning

## Problem

64 pages on the site use kramdown's `{:toc}` directive, which emits an unstyled
`<ul id="markdown-toc">` inline at the top of the content. Several of these pages
are very long:

| Page | Headings | Words |
|---|---:|---:|
| `subpages/books/probabilistic-machine-learning/index.md` | 719 | 127,817 |
| `subpages/books/introduction-to-algorithms/index.md` | 464 | 92,976 |
| `subpages/books/introductionto_lie_groups_taylor/index.md` | 420 | 121,791 |
| `_notes/random.md` | 348 | 57,433 |
| `subpages/books/high_dimensional_data_analysis/index.md` | 241 | 42,646 |

A dozen more sit in the 130–250 heading range.

Two problems follow. First, navigation mid-page is impossible without scrolling
back to the top. Second, the inline contents list is itself an obstacle: on the
largest pages a reader scrolls past several screens of links before reaching the
first paragraph.

The `.toc` CSS block at `assets/css/site.css:388` is dead code — no markup on the
site carries that class. The real TOC (`ul#markdown-toc`) is entirely unstyled.

## Solution

A table-of-contents panel that lives on the right edge of the viewport, revealed
on hover, with the heading tree collapsed to the section currently being read.

### Interaction model

A thin rail is always visible against the right edge, carrying one tick mark per
top-level heading, positioned proportionally to that heading's offset in the
document. The tick for the current section is highlighted, so the collapsed rail
alone answers "where am I".

Hovering the rail — or a ~24px hot-zone along the right edge — slides out the
full panel. It closes roughly 250ms after the pointer leaves, which prevents
flicker when the cursor crosses a corner.

Pinning is width-driven only. `.wrapper` is 1100px and the panel is ~280px, so
the panel fits in the gutter without overlapping body text at viewport widths of
1660px and above; there it is pinned open. Below that it is a hover overlay.
There is no manual pin toggle: pin state is deliberately not persisted, and a
toggle that forgets its setting on every navigation is worse than no toggle.

### Density

Top-level headings are always listed. The branch containing the active heading
auto-expands; other branches stay collapsed. Clicking a disclosure triangle
overrides the automatic collapse for that branch until the reader navigates to a
different section. On the 719-heading page this yields roughly a dozen visible
rows instead of 719.

A filter field at the top of the panel narrows the whole tree as the reader
types, which supplies fuzzy-jump speed without a second piece of UI.

Clicking a heading scrolls to it. In hover-overlay and drawer modes the panel
then closes, since it is covering the text being navigated to; in pinned mode it
stays open.

### The existing inline TOC

Once the script runs, `#markdown-toc` is **reparented** into the panel and no
longer appears inline. Content now starts immediately on every page.

Reparenting rather than cloning is a deliberate choice. Headings on these pages
contain math (`$V^\pi$` and similar), and a clone would force MathJax to typeset
a second copy of all 719 rows on the heaviest page. Moving the same nodes costs
nothing.

If the script does not run — disabled JS, load failure — nothing is moved and
every page renders exactly as it does today. That is the fallback, and it
requires no extra code.

### Narrow viewports

Below 900px (an existing breakpoint in `site.css`) the rail is hidden and a small
floating contents button sits in the bottom-right, within thumb reach. Tapping it
opens the same accordion-and-filter panel as a full-height drawer over a dimmed
backdrop. Tapping a heading or the backdrop closes it. Same component, same code
— only the trigger differs.

### Keyboard access

`/` or `Cmd/Ctrl+K` opens the panel with the filter focused. The shortcut is
suppressed while focus is inside an input or textarea, so the existing `/search/`
page is unaffected. `↑`/`↓` walk the visible rows, `Enter` jumps, `Esc` closes.

The panel carries `role="navigation"` and an `aria-label`; the rail exposes
`aria-expanded`. This is also what makes the feature usable by anyone who cannot
hover at all.

### Heading anchors

A faint `¶` appears beside any `h1`–`h3` carrying an id when the heading is
hovered. Clicking it copies the absolute deep link (e.g.
`…/index.html#value-iteration`) to the clipboard and shows a brief confirmation.
Where `navigator.clipboard` is unavailable the element behaves as an ordinary
link to the fragment. kramdown already generates the ids, so nothing changes
server-side.

## Architecture

Progressive enhancement, entirely client-side. All 64 TOC-bearing pages use
`layout: default`, so there is a single integration point.

| File | Change |
|---|---|
| `assets/js/toc-panel.js` | new; vanilla JS, no dependencies, target ≤14 KB unminified |
| `assets/js/heading-anchors.js` | new; the `¶` copy-link affordance, independent of the panel |
| `assets/css/site.css` | new section appended; extends the `prefers-reduced-motion` block at line 236 |
| `_layouts/default.html` | two `<script defer>` tags, one per file |

No markdown files are edited. No Jekyll plugins are added — GitHub Pages would
reject them.

The script exits immediately when the page has no `ul#markdown-toc`, or when the
TOC holds fewer than 4 entries, where a panel would be noise.

`.site-header` is `position: static` and scrolls away, so the panel is
`position: fixed` at full height with no top-offset arithmetic.

## Components

Six units across two files. Each owns exactly one concern, and each can be
understood without reading the others.

**`parseToc()`** — the only unit that knows kramdown's markup contract. Reads
`ul#markdown-toc` and returns `[{id, text, level, headingEl}]`. If kramdown's
output ever changes, this is the only function that needs to change.

**`renderPanel(model)`** — builds the rail and panel DOM from the model. A pure
function of its input; knows nothing about scrolling or hovering.

**`createScrollSpy(model, onChange)`** — cached heading offsets plus a
rAF-throttled scroll handler, not an `IntersectionObserver`. Sections in these
notes routinely run several screens tall, so a 20vh intersection band would go
unintersected for most of a scroll and the active heading would blank out
instead of tracking the reader. The offsets are recomputed on load, on resize,
and on a `ResizeObserver` reflow (MathJax re-typesets well after page load, with
no scroll event of its own). Sole owner of "where am I"; emits an id.

**`createRevealController()`** — sole owner of open/closed state: the hover zone,
the width-driven pin, the mobile drawer, and Esc.

**`createFilter(model, listEl)`** — substring match against two indexes stored per
row: the raw heading text, and a math-stripped variant with `$` and TeX control
sequences removed, so typing `V pi` matches a heading rendered as `$V^\pi$`.

**`createAnchors()`** — the `¶` links. Fully independent of the panel; deleting it
touches nothing else. Ships as its own file, `assets/js/heading-anchors.js`,
rather than inside `toc-panel.js`, precisely because it is independent.

## Styling

The panel uses the existing custom properties — `--surface`, `--surface-muted`,
`--line`, `--muted`, `--accent`, `--accent-muted`, `--shadow` — so dark mode
works with no theme-specific rules.

`@media print` hides the rail, floating button and anchors. Because the heading
list has been physically moved into the panel, it cannot be "restored" by CSS
alone; instead the panel itself drops its fixed positioning and prints as an
ordinary block at the top of the document, with the accordion fully expanded and
the filter field hidden. On a page where the script did not run, the inline TOC
is untouched and prints as it does today.

`@media (prefers-reduced-motion: reduce)` disables the slide transition.

## Verification

The repository has no test framework; it is a Jekyll site. Verification is a
local build via the project's existing convention:

```
BUNDLE_GEMFILE=Gemfile.local bundle exec jekyll serve
```

followed by a manual matrix.

Pages to check:

- `subpages/books/probabilistic-machine-learning/index.md` — 719 headings, the
  stress case for accordion collapse and MathJax cost
- `subpages/books/reinforcement_learning_overview/index.md` — 182 headings,
  contains math in headings
- a small TOC page, to confirm the panel is proportionate
- a page with no `{:toc}`, to confirm the script no-ops

Axes to check on those pages:

- light and dark theme
- viewport ≥1660px (pinned), 1100–1660px (hover overlay), <900px (drawer)
- JavaScript disabled — inline TOC must still render as it does today
- headings containing math must be legible in the panel and matchable in the
  filter
- full keyboard path: `/`, `Cmd/Ctrl+K`, `↑`, `↓`, `Enter`, `Esc`, and sane Tab
  order
- `/search/` page — confirm `/` still types into the search field rather than
  opening the panel

## Out of scope

- Any edit to the 64 markdown files
- Site-wide navigation changes beyond the two script tags
- Persisting pin state across pages
- A reading-progress fill on the rail
- New Jekyll plugins
- Removing the dead `.toc` CSS block at `site.css:388` — unrelated cleanup
