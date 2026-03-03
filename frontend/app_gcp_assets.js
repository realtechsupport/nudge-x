// app_gcp_assets.js
// Small helper module for GCP-related constants & asset URL generation.
// Used by App.jsx / app.js.

export const GcpRegion = "us-central1"; // adjust to your actual region if needed

// Base URL for your public GCS bucket (no trailing slash is also fine).
const GCP_BUCKET_BASE_URL = "https://storage.googleapis.com/nudge-bucket";

/**
 * Given a relative asset path (e.g. "site-logo.png" or "icons/pin.svg"),
 * returns the full URL in the GCS bucket.
 */
export function getAssetUrl(path) {
  if (!path) return "";
  // ensure exactly one slash between base and path
  if (path.startsWith("/")) {
    return `${GCP_BUCKET_BASE_URL}${path}`;
  }
  return `${GCP_BUCKET_BASE_URL}/${path}`;
}
