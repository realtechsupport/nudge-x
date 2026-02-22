
import React, { useRef, useEffect, useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from "react-router-dom";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { WireframeGeometry2 } from "three/examples/jsm/lines/WireframeGeometry2.js";
import { Wireframe } from "three/examples/jsm/lines/Wireframe.js";
import SiteDisplay from "./SiteDisplay.jsx";

import { GcpRegion } from "./app_gcp_assets";
import {
  fetchSitesFromTsv,
  getColorForSeverity,
  latLonToVector3,
} from "./siteData.js";


function Globe({ autoSiteIndex, setAutoSiteIndex }) {
  const AUTO_MINE_NAMES = [
    "ShiluMine",
    "SerraAzulMine",
    "GarzweilerOpenPitMine",
    "SprucePineQuarry",
    "YallournOpenCut",
  ];
  const mountRef = useRef(null);
  const navigate = useNavigate();

  const [siteData, setSiteData] = useState([]);
  const [sitesLoading, setSitesLoading] = useState(true);
  const [sitesError, setSitesError] = useState(null);

  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const clickableObjectsRef = useRef([]);
  const autoTriggerRef = useRef(false);
  const autoSitesRef = useRef([]);
  const animationFrameRef = useRef(null);
  const rotationPausedRef = useRef(false);

  useEffect(() => {
    let aborted = false;

    async function loadSites() {
      try {
        setSitesLoading(true);
        setSitesError(null);
        const sites = await fetchSitesFromTsv();
        if (!aborted) {
          setSiteData(sites);
        }
      } catch (err) {
        console.error("Failed to load TSV site data:", err);
        if (!aborted) {
          setSitesError(err.message || String(err));
        }
      } finally {
        if (!aborted) {
          setSitesLoading(false);
        }
      }
    }

    loadSites();
    return () => {
      aborted = true;
    };
  }, []);

  useEffect(() => {
    if (!mountRef.current) return;
    if (!siteData || siteData.length === 0) return;
    if (rendererRef.current) return;

    const width = mountRef.current.clientWidth || window.innerWidth;
    const height = mountRef.current.clientHeight || window.innerHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x232323); // rgb(35,35,35)
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.z = 4;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(width, height);
    rendererRef.current = renderer;
    mountRef.current.appendChild(renderer.domElement);

    const globeRadius = 1.5;
    const globeSegments = 32;
    const globeGeometry = new THREE.SphereGeometry(globeRadius, globeSegments, globeSegments);

    let solidGlobe;
    let wire;
    let dotGroup;

    solidGlobe = new THREE.Mesh(
      globeGeometry,
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );

    const loader = new THREE.TextureLoader();
    loader.load(
      "/BlankMap-World-Equirectangular_v3.png",
      (texture) => {
        solidGlobe.material.map = texture;
        solidGlobe.material.needsUpdate = true;
        scene.add(solidGlobe);
        if (wire) scene.add(wire);
        if (dotGroup) scene.add(dotGroup);
      },
      undefined,
      (err) => {
        console.warn("Failed to load earth-continents-black.png:", err);
        // If the texture fails, still show the globe and overlays
        scene.add(solidGlobe);
        if (wire) scene.add(wire);
        if (dotGroup) scene.add(dotGroup);
      }
    );

    const wireMaterial = new LineMaterial({
      color: 0x444444,
      linewidth: 0.0015,
      resolution: new THREE.Vector2(
        width * window.devicePixelRatio,
        height * window.devicePixelRatio
      ),
    });
    wire = new Wireframe(
      new WireframeGeometry2(globeGeometry),
      wireMaterial
    );
    // scene.add(wire); // moved to texture onLoad

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    dotGroup = new THREE.Group();
    const clickableObjects = [];
    const DOT_RADIUS = 0.025;
    const DOT_WIDTH_SEGMENTS = 16;
    const DOT_HEIGHT_SEGMENTS = 16;
    const dotGeometry = new THREE.SphereGeometry(
      DOT_RADIUS,
      DOT_WIDTH_SEGMENTS,
      DOT_HEIGHT_SEGMENTS
    );
    // Flatten & shrink: 30% smaller diameter (X/Y) and 15% thickness along radial (Z)
    dotGeometry.scale(0.66, 0.66, 0.25);

    siteData.forEach((site) => {
      const pos = latLonToVector3(site.lat, site.lon, globeRadius);

      const baseColor = getColorForSeverity(site.severity);
      const mat = new THREE.MeshBasicMaterial({
        color: baseColor,
        transparent: true,
        opacity: 0.6,
      });
      const dot = new THREE.Mesh(dotGeometry, mat);
      dot.position.copy(pos);
      dot.userData = site;
      dot.lookAt(0, 0, 0);

      // Determine if this site should be auto-triggered based on mine name list
      const labelCombined = `${site.name || ""} ${site.mine || ""} ${site.mine_name || ""}`.toLowerCase();
      const matchedIndex = AUTO_MINE_NAMES.findIndex((m) =>
        labelCombined.includes(m.toLowerCase())
      );
      if (matchedIndex !== -1) {
        const basePos = pos.clone();
        const triggerAngle = Math.atan2(-basePos.x, basePos.z);
        autoSitesRef.current.push({
          dot,
          site,
          triggerAngle,
          autoIndex: matchedIndex,
        });
      }

      dotGroup.add(dot);
      clickableObjects.push(dot);
    });

    clickableObjectsRef.current = clickableObjects;
    // scene.add(dotGroup); // moved to texture onLoad

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onCanvasClick = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(clickableObjectsRef.current, true);
      if (intersects.length > 0) {
        const selected = intersects[0].object.userData;
        console.log("Clicked site:", selected);
        navigate(`/site/${encodeURIComponent(selected.name)}`, {
          state: selected,
        });
      }
    };

    renderer.domElement.addEventListener("click", onCanvasClick);

    const onWindowResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth || window.innerWidth;
      const h = mountRef.current.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      if (wireMaterial && wireMaterial.resolution) {
        wireMaterial.resolution.set(
          w * window.devicePixelRatio,
          h * window.devicePixelRatio
        );
      }
    };
    window.addEventListener("resize", onWindowResize);

    const tempAngleState = { lastAngle: null };

    const checkAutoTrigger = () => {
      if (autoTriggerRef.current) return;
      const targets = autoSitesRef.current;
      if (!targets || targets.length === 0) return;

      const twoPi = Math.PI * 2;
      const current = ((solidGlobe.rotation.y % twoPi) + twoPi) % twoPi;
      const offset = 0.0; // no extra phase offset to avoid misalignment

      const targetIndex = typeof autoSiteIndex === "number" ? autoSiteIndex : 0;

      let chosenEntry = null;
      let chosenDiff = Infinity;

      for (const entry of targets) {
        const { dot, site, triggerAngle, autoIndex } = entry;
        if (autoIndex !== targetIndex) continue;
        if (!dot || !dot.material || !dot.material.color) continue;

        const target = (((triggerAngle + offset) % twoPi) + twoPi) % twoPi;
        let diff = Math.abs(current - target);
        diff = Math.min(diff, twoPi - diff);

        if (diff < chosenDiff) {
          chosenDiff = diff;
          chosenEntry = entry;
        }
      }

      // Use a slightly looser threshold so the trigger actually fires
      const threshold = 0.08; // ~4.6 degrees
      if (!chosenEntry || chosenDiff >= threshold) return;

      const { dot, site } = chosenEntry;

      autoTriggerRef.current = true;
      rotationPausedRef.current = true;

      // Advance to the next auto site for the next globe session
      if (typeof setAutoSiteIndex === "function" && AUTO_MINE_NAMES.length > 0) {
        setAutoSiteIndex((prev) => (prev + 1) % AUTO_MINE_NAMES.length);
      }

      const label = site.name || site.mine || "Selected Site";

      const originalColor = dot.material.color.getHex();
      const blinkOnColor = 0x0000ff;     //white: 0xffffff;
      const blinkDurationMs = 2000;
      const blinkIntervalMs = 125;
      let elapsed = 0;
      let isOn = false;

      const intervalId = setInterval(() => {
        isOn = !isOn;
        dot.material.color.set(isOn ? blinkOnColor : originalColor);
        elapsed += blinkIntervalMs;

        if (elapsed >= blinkDurationMs) {
          clearInterval(intervalId);
          dot.material.color.set(originalColor);

          requestAnimationFrame(() => {
            navigate(`/site/${encodeURIComponent(label)}`, {
              state: site,
            });
          });
        }
      }, blinkIntervalMs);
    };

    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      const rotationSpeed = 0.0005;
      if (!rotationPausedRef.current) {
        solidGlobe.rotation.y += rotationSpeed;
      }
      wire.rotation.y = solidGlobe.rotation.y;
      dotGroup.rotation.y = solidGlobe.rotation.y;

      // Check if any auto-target mine is near its trigger angle
      checkAutoTrigger();

      controls.update();
      renderer.render(scene, camera);
    };
    animate();;

    return () => {
      cancelAnimationFrame(animationFrameRef.current);
      renderer.domElement.removeEventListener("click", onCanvasClick);
      window.removeEventListener("resize", onWindowResize);
      controls.dispose();
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose && m.dispose());
          } else if (obj.material.dispose) {
            obj.material.dispose();
          }
        }
      });
      renderer.dispose();
      if (
        mountRef.current &&
        renderer.domElement &&
        renderer.domElement.parentNode === mountRef.current
      ) {
        mountRef.current.removeChild(renderer.domElement);
      }
      rendererRef.current = null;
    };
  }, [siteData, navigate]);

  return (
    <div className="globe-wrapper">
      <div ref={mountRef} className="globe-canvas" />
      {sitesLoading && <div className="status-text">Loading site data from TSV...</div>}
      {!sitesLoading && sitesError && <div className="status-text error">Failed to load site data: {sitesError}</div>}
    </div>
  );
}

export default function App() {
  const [autoSiteIndex, setAutoSiteIndex] = useState(0);

  return (
    <div className="w-full min-h-screen bg-black text-white">
      <Router>
        <Routes>
          <Route
            path="/"
            element={
              <Globe
                autoSiteIndex={autoSiteIndex}
                setAutoSiteIndex={setAutoSiteIndex}
              />
            }
          />
          <Route path="/site/:name" element={<SiteDisplay />} />
        </Routes>
      </Router>
    </div>
  );
}
