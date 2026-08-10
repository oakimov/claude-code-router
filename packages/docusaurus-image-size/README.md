# Docusaurus image-size adapter

This private workspace package replaces Docusaurus's archived `image-size`
dependency. It preserves the `image-size/fromFile` API used by
`@docusaurus/mdx-loader` while delegating local stream parsing to the maintained
`probe-image-size` package. It is documentation-build tooling only and is not
included in the CCR runtime or published packages.
