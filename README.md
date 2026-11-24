# sirotenkogeorgii.github.io

Personal site: About, Publications, Research, Blog (and CV/Notes links).
Plain Jekyll 4, no theme. Deployed by `.github/workflows/pages.yml`
(GitHub Pages source: "GitHub Actions").

## Local preview

    bundle install
    bundle exec jekyll serve      # http://127.0.0.1:4000
    bundle exec jekyll build && bundle exec htmlproofer _site --disable-external=true

## Where things live

| What | File |
|---|---|
| About (landing) | `index.md` |
| Publications | `_data/publications.yml` (rendered by `publications.html`) |
| Research | `research.md` (fill in the `TODO` comments) |
| Blog posts | `_posts/YYYY-MM-DD-title.md` (MathJax on by default) |
| CV | put the PDF at `assets/cv.pdf`; the nav link appears automatically |
| Notes link | set `notes_public: true` in `_config.yml` once a note is public at `/notes/` |
| Old note URLs | `_redirects/*.md` with `permalink:` (old URL) and `redirect_to:` (new URL) |
| Landing figure | `_scripts/make_flow_figure.py` writes `_includes/flow-figure.svg` |
| Styles | `assets/css/site.css` |
