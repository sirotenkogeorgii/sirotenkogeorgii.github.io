---
layout: default
title: Home
---


- [Notes]({{ '/notes/' | relative_url }}) — in-depth write-ups and study notes
- [Tags]({{ '/tags/' | relative_url }}) — every page grouped by topic, with a filterable tag cloud
- [Graph]({{ '/graph/' | relative_url }}) — an interactive map of how the topics connect
- [Search]({{ '/search/' | relative_url }}) — full-text search across all pages
<!-- - [GitHub](https://github.com/) — source code and experiments -->
<!-- - [LinkedIn](https://www.linkedin.com/) — professional updates -->

{% if site.data.recent_updates and site.data.recent_updates.entries %}
## Recent updates

<ul class="recent-updates">
  {% for entry in site.data.recent_updates.entries limit: 10 %}
  <li>
    <time class="recent-updates__date" datetime="{{ entry.date }}">{{ entry.date }}</time>
    <a href="{{ entry.url | relative_url }}">{{ entry.title }}</a>
    {% if entry.added %}<span class="recent-updates__badge">new</span>{% endif %}
  </li>
  {% endfor %}
</ul>
{% endif %}