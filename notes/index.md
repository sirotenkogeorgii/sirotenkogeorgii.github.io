---
layout: default
title: Notes
permalink: /notes/
---

# Notes

Long-form write-ups and study notes. Subscribe to stay in the loop:

- <a href="{{ '/feed.xml' | relative_url }}">RSS feed</a>

{% assign notes = site.notes | sort: 'date' | reverse %}
{% if notes.size > 0 %}
<ul class="note-list">
  {% for note in notes %}
  <li>
    <a href="{{ note.url | relative_url }}">{{ note.title }}</a>
    {% if note.date %}<span class="note-list__meta">— {{ note.date | date: '%B %-d, %Y' }}</span>{% endif %}
    {% if note.excerpt %}<p>{{ note.excerpt | strip_html }}</p>{% endif %}
  </li>
  {% endfor %}
</ul>
{% else %}
<p>No notes yet. Check back soon!</p>
{% endif %}
