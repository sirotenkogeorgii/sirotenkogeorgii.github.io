---
layout: default
---

<figure class="flow-figure">
{% include flow-figure.svg %}
<figcaption>Nine steps along the Wasserstein geodesic from a bimodal to a unimodal density.</figcaption>
</figure>

# Georgii Sirotenko

I'm an M.Sc. student at Heidelberg University, studying Data and Computer Science and Scientific Computing in parallel.

## Research interests

Mathematical foundations of machine learning: learning theory, information theory, optimization and generative modelling. In particular: what makes learning tractable and which structural properties of learning problems enable efficient algorithms; learning dynamics (compression, flows on spaces of probability measures, the transition from memorization to generalization); and a theory of neural architectures: what networks learn and represent, and how their components shape generalization.

## Contact

{% assign cv = site.static_files | where: "path", "/assets/cv.pdf" | first %}
<ul class="contact">
  <li><a href="mailto:{{ site.author.email }}">{{ site.author.email }}</a></li>
  <li><a href="https://github.com/{{ site.github_username }}">GitHub</a></li>
  {% if cv %}<li><a href="{{ '/assets/cv.pdf' | relative_url }}">CV (PDF)</a></li>{% endif %}
  {% if site.notes_public %}<li><a href="{{ '/notes/' | relative_url }}">Notes</a></li>{% endif %}
</ul>
