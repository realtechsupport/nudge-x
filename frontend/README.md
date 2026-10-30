# Nudge-x globe frontend

The globe displays exported captions and images. It does not run inference,
PostgreSQL, or Qdrant in the browser or on the static hosting service.

## Build

Use Node.js 22.12 or later. From this directory:

```sh
npm ci
npm run build
npm run preview
```

Publish the contents of `dist/`. For a repository site at
`https://USERNAME.codeberg.page/nudge-x/`, build instead with:

```sh
npm run build -- --base=/nudge-x/
```

For a custom domain serving the app at its root, use the default `/` base.
The router, caption data, map texture, and back icon use this build-time base.
Use the same base when previewing a subpath build:
`npm run preview -- --base=/nudge-x/`.

The build writes `_redirects` for `/site/*` routes, including the base path.
Publish that file along with the build when using Codeberg's git-pages service.
For Nginx, retain the SPA fallback described in
[the deployment guide](notes/earth_deployment_instructions.md).
Direct site links reload their data from the exported TSV when navigation
state is unavailable.

## Update the captions

1. Export and clean the captions using the research pipeline.
2. Place the resulting TSV in `public/`.
3. Update the filename in `siteData.js` (`TSV_URL`), preserving
   `import.meta.env.BASE_URL`.
4. Rebuild, check the globe and a directly opened site URL, and publish `dist/`.

## External services and asset rights

Images currently load from the public Google Cloud Storage bucket configured
in `app_gcp_assets.js`. The HTML also loads the Ubuntu font from Google Fonts.
Moving the Git repository or the static build does not move these resources.
See [the licensing inventory](../docs/licensing_inventory.md) before redistributing
assets or migrating the project to another host.

Codeberg Pages reference:
https://docs.codeberg.org/codeberg-pages/advanced-usage/
