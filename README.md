## Nudge-x

<i> Nudge junkyard AI models to do new things.</i>
<br><br>

<p align="center">
<img src="https://github.com/realtechsupport/nudge-x/blob/main/Rotating-Earth.png?raw=true" alt="Mining sitesin project Nudge"/>
</p>
This repository contains the code, documents and documentation required for project NUDGE-x, from the series Scrapyard AI. 
<br>
It implements a Sentinel-2 satellite asset interpretation pipeline that combines statistical operations, human judgement, and generative AI models to interpret the landscape of industrial mining sites across planet Earth.

<br><br>
If you want to rebuild the mining area satellite image interpretation pipeline, follow the 
[installation instructions.](install_instructions_v2.md)
<br>
If you want to recreate the UI, follow the instructions in the
[application deployment.](https://github.com/realtechsupport/nudge-x/blob/main/frontend/notes/earth_deployment_instructions.md)

For a reproducible frontend build and static hosting under a custom domain or
repository subpath, see the [frontend guide](frontend/README.md).

## Research and sources

Tammali, Sai Krishna; Kumar, Vinaya; Böhlen, Marc (2026).
[Synthetic Reflections on Resource Extraction](https://arxiv.org/abs/2602.09299),
arXiv:2602.09299. Sections 2 and 3.2 describe the imagery and metadata sources;
Section 7.2 describes the long-form document used for retrieval.

## License and attribution

Marc Böhlen's original project code is licensed under the [MIT License](LICENSE).
Third-party code and material retain their own terms; this grant does not
relicense datasets, satellite imagery, maps, publications, model outputs, or
other contributors' work without their authorization.

See [source attribution](docs/source_attribution.md) for Copernicus and Wikipedia
credits and [the licensing inventory](docs/licensing_inventory.md) for remaining
asset-specific questions.
