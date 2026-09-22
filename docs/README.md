# TriPAH project website

The official project page is hosted by GitHub Pages from this repository's `main` branch, `/docs` directory:

https://medical-multi-agent-ai.github.io/TriPAH/

Edit `index.html`, `styles.css`, and `app.js` here. Pushing updates to `main` triggers GitHub's Pages build. No package installation or separate hosting account is required. `.nojekyll` preserves the plain static files.

For a local preview from the repository root:

```bash
python -m http.server 4173 --directory docs
```

## Figure sources

- The framework and qualitative retrieval panels come from the authors' supplied paper and figure PDFs. Qualitative panels preserve the original text and labels.
- `assets/quantitative-retrieval.webp` visualizes the means reported in Table I; its values are in `assets/benchmark-means.csv`.
- `assets/quantitative-ablation.webp` visualizes Table IV; its values are in `assets/component-ablation.csv`.
- These are reported results, not a fresh training run. The charts add no new measurements or uncertainty estimates.

## Layout references

The figure carousel advances every six seconds while visible. Hovering pauses playback, and moving the pointer away resumes it automatically, including after a manual arrow click. Left/right arrows and dots select figures. Keyboard focus and touch interaction temporarily pause playback; reduced-motion preferences disable it. All slides share a fixed image viewport and equal card height. On small screens, wide qualitative panels scroll horizontally and portrait charts scroll vertically inside that viewport; full-size links remain available.

The independently implemented HTML/CSS/JavaScript draws layout inspiration from [Nerfies](https://github.com/nerfies/nerfies.github.io) and the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template): centered paper metadata, grouped resource links, wide research figures, and a captioned single-image carousel. Template source files and their demo assets are not bundled.
