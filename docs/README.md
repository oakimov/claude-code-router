# Claude Code Router Documentation

This directory contains the documentation website built with [Docusaurus](https://docusaurus.io/).

## Development

### Install Dependencies

```bash
cd docs
bun install
```

### Start Development Server

```bash
# From docs directory
bun run start

# Or from root directory
bun run dev:docs
```

Open [http://localhost:3000](http://localhost:3000) to view the documentation.

## Build

```bash
# From docs directory
bun run build

# Or from root directory
bun run build:docs
```

The built files will be in the `build/` directory.

## Serve Built Files

```bash
# From docs directory
bun run serve

# Or from root directory
bun run serve:docs
```

## Structure

```
docs/
├── docs/              # Markdown documentation files
│   ├── intro.md       # Introduction page
│   ├── installation.md
│   ├── config/        # Configuration docs
│   ├── advanced/      # Advanced topics
│   └── cli/           # CLI reference
├── src/               # React components and pages
│   ├── components/    # Custom React components
│   ├── pages/         # Additional pages
│   ├── css/           # Custom CSS
│   └── theme/         # Docusaurus theme customization
├── static/            # Static assets (images, etc.)
├── i18n/              # Internationalization files
├── docusaurus.config.ts  # Docusaurus configuration
└── sidebars.ts        # Documentation sidebar structure
```

## Adding Documentation

### Adding New Docs

Create a new Markdown file in the `docs/` directory and add it to `sidebars.ts`.

### Adding New Pages

Add React components to `src/pages/`.

### Customizing Styles

Edit `src/css/custom.css`.

## Internationalization

Documentation supports both English and Chinese.

- English: `docs/` and `src/`
- Chinese: `i18n/zh/docusaurus-plugin-content-docs/current/`

To add Chinese translations:

1. Create corresponding files in `i18n/zh/docusaurus-plugin-content-docs/current/`
2. Translate the content

## Deployment

The documentation can be deployed to:

- GitHub Pages
- Netlify
- Vercel
- Any static hosting service

See [Docusaurus deployment docs](https://docusaurus.io/docs/deployment) for details.
