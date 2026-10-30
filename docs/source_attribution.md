# Sources and attribution

Research reference: Tammali, Sai Krishna; Kumar, Vinaya; Böhlen, Marc (2026),
[Synthetic Reflections on Resource Extraction](https://arxiv.org/abs/2602.09299v2),
arXiv:2602.09299v2. The paper and its bibliography document the research sources;
they do not replace the individual sources' reuse terms.

## Satellite imagery

The maintainer identifies the satellite imagery as EU Copernicus Sentinel-2 data.
The paper's Section 2 describes the imagery and processing, including openEO.
Processed image files in `data/images/` and `sample_images/` carry acquisition
dates in their filenames.

Attribution: **Contains modified Copernicus Sentinel data [year of acquisition].**
For an individual image, use the year in its acquisition-date filename in place
of the bracketed text. Preserve this attribution when redistributing images.

These assets are governed by the
[Copernicus Sentinel data legal notice](https://cds.climate.copernicus.eu/licences/ec-sentinel),
not the project's MIT code license. See also the
[Copernicus citation guidance](https://documentation.dataspace.copernicus.eu/FAQ.html).
Exact product identifiers are not recorded for every image in this repository.

## Mining metadata

Section 3.2 of the paper describes human-compiled site metadata drawing on
Wikipedia, Mindat, company documents, Global Energy Monitor, environmental
agencies, and scholarly sources. The `sources` column of
`data/metadata/Mines_Metadata_v35.tsv` records site-specific links.

Wikipedia contributors are credited through the linked source articles and their
revision histories. Copied or adapted Wikipedia text must retain applicable
attribution and license terms, including ShareAlike where required; see the
[Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).
Individual Wikimedia images can have different licenses and require their own
file-page attribution. Wikipedia sourcing does not establish the license of every
metadata entry or map asset. Other sources retain their own terms.

The metadata's per-entry distinction between original factual summaries and
copied/adapted text remains to be checked. No blanket MIT license is applied to
this mixed-source dataset.

## Long-form context

Section 7.2 and reference 33 identify Benedict Scambary (2013),
*My Country, Mine Country: Indigenous People, Mining and Development Contestation
in Remote Australia*, ANU E Press, DOI
[10.22459/CAEPR33.05.2013](https://doi.org/10.22459/CAEPR33.05.2013).
The paper establishes its research use; permission to redistribute the bundled
PDF remains a separate question. See the licensing inventory.

## Development provenance

The paper documents model inference, evaluation, retrieval, and human review as
research methods. Separately, the maintainer reports that AI coding assistance
focused on refactoring Python notebooks into standalone `.py` files.
This cleanup's frontend fixes and licensing documentation were prepared with
AI assistance and require maintainer review. These descriptions record the
tasks involved, without assigning an unsupported percentage of AI-authored code.
