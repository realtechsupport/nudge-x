
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


function Globe() {
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
  const animationFrameRef = useRef(null);

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
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.z = 4;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(width, height);
    rendererRef.current = renderer;
    mountRef.current.appendChild(renderer.domElement);

    const globeRadius = 2;
    const globeSegments = 32;
    const globeGeometry = new THREE.SphereGeometry(globeRadius, globeSegments, globeSegments);

    const solidGlobe = new THREE.Mesh(
      globeGeometry,
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    scene.add(solidGlobe);

    const loader = new THREE.TextureLoader();
    loader.load(
      "/BlankMap-World-Equirectangular_v3.png",
      (texture) => {
        solidGlobe.material.map = texture;
        solidGlobe.material.needsUpdate = true;
      },
      undefined,
      (err) => {
        console.warn("Failed to load earth-continents-black.png:", err);
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
    const wire = new Wireframe(
      new WireframeGeometry2(globeGeometry),
      wireMaterial
    );
    scene.add(wire);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    const dotGroup = new THREE.Group();
    const clickableObjects = [];
    const dotGeometry = new THREE.SphereGeometry(0.025, 8, 8);

    siteData.forEach((site) => {
      const pos = latLonToVector3(site.lat, site.lon, globeRadius);

      const mat = new THREE.MeshBasicMaterial({
        color: getColorForSeverity(site.severity),
      });
      const dot = new THREE.Mesh(dotGeometry, mat);

      dot.position.copy(pos);
      dot.userData = site;
      dot.lookAt(0, 0, 0);

      dotGroup.add(dot);
      clickableObjects.push(dot);
    });

    clickableObjectsRef.current = clickableObjects;
    scene.add(dotGroup);

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

    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      const rotationSpeed = 0.0005;
      solidGlobe.rotation.y += rotationSpeed;
      wire.rotation.y = solidGlobe.rotation.y;
      dotGroup.rotation.y = solidGlobe.rotation.y;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

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
  return (
    <div className="w-full min-h-screen bg-black text-white">
      <Router>
        <Routes>
          <Route path="/" element={<Globe />} />
          <Route path="/site/:name" element={<SiteDisplay />} />
        </Routes>
      </Router>
    </div>
  );
}
