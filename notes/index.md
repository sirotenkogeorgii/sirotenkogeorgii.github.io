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
  <li class="note-list__item">
    {% if note.date %}<span class="note-list__date">{{ note.date | date: '%d %b, %Y' }}</span>{% endif %}
    <a class="note-list__title" href="{{ note.url | relative_url }}">{{ note.title }}</a>
  </li>
  {% endfor %}
</ul>
{% else %}
<p>No notes yet. Check back soon!</p>
{% endif %}
