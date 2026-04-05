# Presentation Decks

Reveal.js 5.1.0 slide decks for each module. See `workspaces/decks/briefs/deck-brief.md` for design specifications.

## Local Preview

```bash
cd decks
python -m http.server 8080
# Open http://localhost:8080/ascent01/deck.html
```

## PDF Export

Append `?print-pdf` to any deck URL, then print to PDF from Chrome.

## Structure

```
decks/
├── assets/css/theme.css     # Shared Reveal.js theme
├── assets/img/              # Shared images
├── ascent01/deck.html          # Module 1 slides
├── ascent02/deck.html          # Module 2 slides
├── ...
└── ascent06/deck.html          # Module 6 slides
```
