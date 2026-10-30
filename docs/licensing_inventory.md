# Licensing and provenance inventory

The maintainer selected MIT for his own code on 2026-10-30; see `LICENSE`.
This inventory records the scope and remaining provenance work. It does not
declare all third-party material cleared for Codeberg migration. The research
sources are documented in [source attribution](source_attribution.md) and the
[project paper](https://arxiv.org/abs/2602.09299v2).

| Material | Evidence in this repository | Remaining work |
| --- | --- | --- |
| Original Python code and scripts | MIT selected for Marc Böhlen's own code; license text added | Confirm coverage of any separately owned collaborator contributions |
| Frontend code | Package declaration and lockfile aligned to MIT | Preserve any separately applicable third-party notices |
| Documentation and notebook text | No explicit overall license | Confirm the intended license; notebook outputs and embedded third-party content need separate consideration |
| Satellite images in `data/images/` and `sample_images/` | Maintainer confirms Copernicus Sentinel-2; paper Section 2; attribution and legal-notice link added | Preserve acquisition-year credit; add exact product identifiers where available |
| `data/metadata/Mines_Metadata_v35.tsv` | Paper Section 3.2 describes human-compiled metadata; per-row `sources` column cites multiple websites | Distinguish original summaries from copied/adapted material and retain applicable source terms |
| Exported caption TSVs | Project-generated model outputs and metadata | Document generation/provenance and relevant source/provider terms; do not assume a code license covers all content |
| `data/Scambary_MyCountryMyMine_2013_239p.pdf` | Third-party publication by Benedict Scambary; publisher and DOI identified below | Confirm the bundled edition's redistribution terms; otherwise replace the bundled copy with retrieval instructions |
| `frontend/public/BlankMap-World-Equirectangular_v3.png` | Maintainer confirms the Wikimedia Commons source identified below; source map released under CC0 1.0 | Record any project-specific modifications; source permits modification and redistribution |
| `frontend/public/globe_back.png` | Original homemade icon by Marc Böhlen | State the intended reuse license for this image |
| `Rotating-Earth.png` | Screenshot from a video, as confirmed by the maintainer | Confirm video ownership and intended reuse license |
| Images served from Google Cloud Storage | Publicly accessible Copernicus Sentinel-2 images processed by Marc Böhlen; provenance confirmed by the maintainer | Preserve Copernicus attribution, including the acquisition year, under the same terms as the repository's Sentinel-2 imagery |

Do not apply a new project license to third-party material without checking its
existing terms. Preserve license notices for dependencies and upstream material.

## Confirmed map source

The maintainer identifies
[BlankMap-World-Equirectangular.svg on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:BlankMap-World-Equirectangular.svg)
as the source of `frontend/public/BlankMap-World-Equirectangular_v3.png`.

The Commons page credits John Harvey and upstream contributors, with derivation
from maps based on the CIA World Factbook. It identifies the source map as
available under the
[CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).

The source permits copying, modification and redistribution, including commercial
use, without seeking permission. This source credit is retained for provenance.
The project's PNG version should not be described as an unmodified original SVG.

## Bundled publication

Scambary's *My Country, Mine Country: Indigenous People, Mining and Development
Contestation in Remote Australia* is listed by
[ANU Press](https://press.anu.edu.au/publications/series/caepr/my-country-mine-country),
DOI `10.22459/CAEPR33.05.2013`.

ANU Press's [FAQ](https://press.anu.edu.au/faqs) asks readers to check each book's
copyright page and encourages linking rather than offsite hosting. The bundled
edition still needs to be checked; availability as a free download is not itself
a redistribution license.

## Development provenance

The repository and paper document model inference as part of the research method.
That is separate from the use of AI assistance to write or maintain the code.
The maintainer reports that coding assistance focused on refactoring Python
notebooks into standalone `.py` files. This describes the assistance's purpose;
it does not quantify the proportion of AI-authored code or determine eligibility.
The present cleanup was prepared with AI assistance, as recorded in
`source_attribution.md`.
