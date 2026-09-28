# Nib landing page → `nib-app.pages.dev`

A self-contained static site for Nib. **No build step, no dependencies, no
JavaScript, no third-party requests** — just `index.html` + `styles.css` +
`favicon.svg`, served as-is. This matches Nib's own "vanilla, no-tracking"
ethos and lets the Content-Security-Policy (`_headers`) stay strict.

```
site/
├── index.html     # the page
├── styles.css     # all styling (no inline styles, so CSP needs no 'unsafe-inline')
├── favicon.svg
├── _headers       # Cloudflare Pages security headers (strict CSP)
└── README.md      # this file
```

Preview locally with any static server, e.g.:

```bash
python3 -m http.server -d site 8000   # then open http://localhost:8000
```

## Deploy to Cloudflare Pages

The `nib-app.pages.dev` subdomain is assigned automatically when the Pages
**project name** is `nib-app`. Two ways:

### A. Git integration (recommended — auto-deploys on push)

1. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** →
   **Connect to Git**.
2. Pick this repo. Set **Project name** = `nib-app` → you get `https://nib-app.pages.dev`.
3. Build settings:
   - **Framework preset:** None
   - **Build command:** *(leave empty)*
   - **Build output directory:** `site`
4. **Save and Deploy.** Every push to the production branch redeploys.

### B. Wrangler CLI (one-off / scripted)

```bash
npm i -g wrangler
wrangler login                         # or set CLOUDFLARE_API_TOKEN
wrangler pages project create nib-app  # claims nib-app.pages.dev
wrangler pages deploy site --project-name nib-app
```

## After it's live

- Set the GitHub repo **Website** field (About → gear) to `https://nib-app.pages.dev`.
- A custom domain (e.g. `nib.app`) can be attached later under the project's
  **Custom domains** tab.

> The outbound GitHub links point at `github.com/abgnydn/nib` — the repo's
> current name after the rename.
