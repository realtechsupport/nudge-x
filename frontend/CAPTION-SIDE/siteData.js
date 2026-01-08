
// siteData_Dec03_1500.js
// Data preparation utilities for the TSV-driven globe.

import * as THREE from "three";
import { getAssetUrl } from "./app_gcp_assets";

// Path (from the web root) to the TSV file. Place the TSV in your Vite `public/` folder.
export const TSV_URL = "/frontend_captions_2026-01-08_val.tsv";

/* ----------------- Utility: severity color ----------------- */
export const getColorForSeverity = (severity) => {
  switch (severity) {
    case 5:
      return 0xff0000;
    case 4:
      return 0xff4500;
    case 3:
      return 0xffa500;
    case 2:
      return 0xffff00;
    case 1:
    default:
      return 0x00ff00;
  }
};

// Very small TSV parser for `id, filename, minename, site_location, country, gps_coordinates, caption`
export function parseTsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];

  const headers = lines[0].split("\t");
  return lines.slice(1).map((line) => {
    const cols = line.split("\t");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = cols[i] !== undefined ? cols[i] : "";
    });
    return row;
  });
}

/**
 * Convert TSV rows into normalized `site` objects used by the globe.
 */
export function rowsToSites(rows) {
  return rows
    .map((row) => {
      const coord = (row.gps_coordinates || "").trim();
      if (!coord || coord.toLowerCase() === "unknown") {
        return null; // skip rows without coordinates
      }
      const [latStr, lonStr] = coord.split(",");
      const lat = parseFloat(latStr);
      const lon = parseFloat(lonStr);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        return null;
      }

      return {
        id: row.id,
        name: row.site_location,
	mine: row.mine_name,
        country: row.country,
        lat,
        lon,
        severity: 3, // default; can adjust later or derive from data
        captions: [row.caption],
        // image URL derived from filename via GCS helper
        image: row.filename ? getAssetUrl(row.filename) : null,
        filename: row.filename,
      };
    })
    .filter(Boolean);
}

/**
 * Convenience function: fetch and parse TSV into site objects.
 */
export async function fetchSitesFromTsv() {
  const res = await fetch(TSV_URL, { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`Failed to fetch TSV: ${res.status} ${res.statusText}`);
  }
  const text = await res.text();
  const rows = parseTsv(text);
  return rowsToSites(rows);
}

// Global longitude offset (degrees) to fine-tune texture alignment.
// If you still see a consistent east/west shift, adjust this value:
//   - Negative values move dots west relative to the map.
//   - Positive values move dots east.
export const LONGITUDE_OFFSET_DEG = -9;

/* -------------- Lat/Lon → 3D sphere position ----------------
   lat: degrees north (+) / south (-)
   lon: degrees east (+) / west (-)
   radius: globe radius
   This matches THREE.SphereGeometry's equirectangular UV mapping.
---------------------------------------------------------------- */
export function latLonToVector3(lat, lon, radius) {
  const phi = THREE.MathUtils.degToRad(90 - lat);                    // from north pole down
  const theta = THREE.MathUtils.degToRad(lon + 180 + LONGITUDE_OFFSET_DEG); // -180..180 → 0..360

  const x = -radius * Math.sin(phi) * Math.cos(theta);
  const y =  radius * Math.cos(phi);
  const z =  radius * Math.sin(phi) * Math.sin(theta);

  return new THREE.Vector3(x, y, z);
}
