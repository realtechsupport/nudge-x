// app/src/app.js
import React, { useRef, useEffect, useState, useCallback } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  useParams,
  useLocation,
} from "react-router-dom";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { WireframeGeometry2 } from "three/examples/jsm/lines/WireframeGeometry2.js";
import { Wireframe } from "three/examples/jsm/lines/Wireframe.js";

/* ---------------- STATIC SITE METADATA ---------------- */
/* Keep captions hard-coded here; images are attached async from backend. */
const baseSiteData = [
  {
    name: "AidyrlaGoldDeposit",
    lat: 18.9582,
    lon: 72.8321,
    severity: 5,
    captions: ["The primary hazard is a large, active wildfire within a forested area..."],
  },
  {
    name: "Amable Pit",
    lat: -29.37833,
    lon: -69.94583,
    severity: 5,
    captions: ["The primary hazard is a large, active wildfire within a forested area..."],
  },
  {
    name: "Argyle Diamond Mine",
    lat: -16.71639,
    lon: 128.38889,
    severity: 4,
    captions: ["A significant oil spill is present in the marine environment."],
  },
  {
    name: "Avon North Open Pit",
    lat: -32.11833,
    lon: 151.97861,
    severity: 3,
    captions: ["This satellite image shows Avon North Open Pit."],
  },
  {
    name: "Bengalla Open-Cut Mine",
    lat: -32.26953,
    lon: 150.84403,
    severity: 2,
    captions: ["This satellite image shows Bengalla Open-Cut Mine."],
  }
];

/* ----------------- Utility: severity color ----------------- */
const getColorForSeverity = (severity) => {
  // Map severity to color (hex)
  // 1 = greenish, 5 = red/orange
  switch (severity) {
   
    case 1:
    default:
      return 0xffa500;
  }
};

/* -------------------- MAIN GLOBE COMPONENT -------------------- */
function Globe() {
  const mountRef = useRef(null);
  const navigate = useNavigate();

  // siteData holds the base static site metadata + image url when available.
  const [siteData, setSiteData] = useState(baseSiteData);
  const [imagesLoaded, setImagesLoaded] = useState(false);

  // Refs to hold Three objects that survive re-renders:
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const clickableObjectsRef = useRef([]); // array of Meshes for raycasting
  const animationFrameRef = useRef(null);

  /* ----------------- Async: fetch GCS images (first 5) ----------------- */
  useEffect(() => {
  let aborted = false;

  async function loadImagesFromBackend() {
    try {
      const res = await fetch("http://35.208.244.178:8000/images", { cache: "no-cache" });
      if (!res.ok) {
        console.error("Images endpoint failed:", res.status, await res.text());
        return;
      }
      const images = await res.json();
      console.log("backend images:", images);

      // Build a map from filename (lowercase) => url
      const nameToUrl = {};
      images.forEach((it) => {
        if (it && it.name && it.url) {
          nameToUrl[it.name.toLowerCase()] = it.url;
          // also map without underscores/dates: e.g. "AidyrlaGoldDeposit_rgb_2024-06-02.png" -> "aidyrlagoldd..." + remove dates
          const simple = it.name.replace(/\.\w+$/, "").toLowerCase();
          nameToUrl[simple] = it.url;
        }
      });

      // try to match site by name tokens (loose)
      const mapped = baseSiteData.map((site) => {
        const key1 = site.name.replace(/\s+/g, "").toLowerCase(); // "AmablePit" -> amablepit
        const key2 = site.name.toLowerCase(); // "amable pit"
        // find by exact filename, by simplified name, or fallback to first available image
        const url =
          nameToUrl[`${site.name.toLowerCase()}.png`] ||
          nameToUrl[`${site.name.toLowerCase()}.jpg`] ||
          nameToUrl[key1] ||
          nameToUrl[key2] ||
          images[0]?.url ||
          null;
        return { ...site, image: url };
      });

      if (!aborted) {
        setSiteData(mapped);
        setImagesLoaded(true);
      }
    } catch (err) {
      console.error("Failed to fetch images from backend:", err);
    }
  }

  loadImagesFromBackend();
  return () => {
    aborted = true;
  };
}, []);


  /* ----------------- Update dots' userData when imagesLoaded or siteData change ----------------- */
  useEffect(() => {
    if (!imagesLoaded) return;
    // Attach updated site info (including image) to already-created clickable objects
    const dots = clickableObjectsRef.current;
    if (!dots || dots.length === 0) return;

    dots.forEach((dot) => {
      const match = siteData.find((s) => s.name === dot.userData.name);
      if (match) {
        dot.userData = match;
      }
    });
  }, [imagesLoaded, siteData]);

  /* ----------------- THREE scene initialization (run ONCE) ----------------- */
  useEffect(() => {
    if (!mountRef.current) return;

    // Basic setup
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x282c34);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(width, height);
    rendererRef.current = renderer;
    mountRef.current.appendChild(renderer.domElement);

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dLight.position.set(5, 3, 5);
    scene.add(dLight);

    // Globe
    const globeRadius = 2;
    const globeGeometry = new THREE.SphereGeometry(globeRadius, 64, 64);

    const solidGlobe = new THREE.Mesh(
      globeGeometry,
      new THREE.MeshPhongMaterial({ color: 0x336699 })
    );
    scene.add(solidGlobe);

    // Load texture from /public/earth-dark.jpg — ensure file is present in public/
    const loader = new THREE.TextureLoader();
    loader.load(
      "/earth-dark.jpg",
      (texture) => {
        // apply texture
        solidGlobe.material = new THREE.MeshPhongMaterial({ map: texture });
        solidGlobe.material.needsUpdate = true;
      },
      undefined,
      (err) => {
        console.warn("Failed to load earth-dark.jpg:", err);
        // fallback color already set; continue
      }
    );

 
    // Wireframe
    const wireMaterial = new LineMaterial({
      color: 0x808080,
      linewidth: 0.002,
      resolution: new THREE.Vector2(width * window.devicePixelRatio, height * window.devicePixelRatio),
    });
    const wire = new Wireframe(new WireframeGeometry2(globeGeometry), wireMaterial);
    scene.add(wire);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;

    // ---- Add Site Dots (from baseSiteData so globe appears immediately) ----
    const dotGroup = new THREE.Group();
    const clickableObjects = [];
    const dotGeometry = new THREE.SphereGeometry(0.03, 12, 12);

    baseSiteData.forEach((site) => {
      const lat = (Math.PI / 180) * site.lat;
      const lon = -(Math.PI / 180) * site.lon;

      const x = globeRadius * Math.cos(lat) * Math.cos(lon);
      const y = globeRadius * Math.sin(lat);
      const z = globeRadius * Math.cos(lat) * Math.sin(lon);

      const mat = new THREE.MeshBasicMaterial({ color: getColorForSeverity(site.severity) });
      const dot = new THREE.Mesh(dotGeometry, mat);

      dot.position.set(x, y, z);
      dot.userData = site; // will be replaced/extended later when imagesLoaded
      dot.lookAt(0, 0, 0);

      dotGroup.add(dot);
      clickableObjects.push(dot);
    });

    clickableObjectsRef.current = clickableObjects;
    scene.add(dotGroup);

    // Raycaster for clicks
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
        navigate(`/site/${encodeURIComponent(selected.name)}`, { state: selected });
      }
    };

    renderer.domElement.addEventListener("click", onCanvasClick);

    // Resize handling
    const onWindowResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      if (wireMaterial && wireMaterial.resolution) {
        wireMaterial.resolution.set(w * window.devicePixelRatio, h * window.devicePixelRatio);
      }
    };
    window.addEventListener("resize", onWindowResize);

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      solidGlobe.rotation.y += 0.0008;
      wire.rotation.y += 0.0008;
      dotGroup.rotation.y += 0.0006;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup when component unmounts
    return () => {
      cancelAnimationFrame(animationFrameRef.current);
      renderer.domElement.removeEventListener("click", onCanvasClick);
      window.removeEventListener("resize", onWindowResize);
      controls.dispose();
      // Dispose geometries and materials
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose && m.dispose());
          } else {
            obj.material.dispose && obj.material.dispose();
          }
        }
      });
      renderer.dispose();
      if (mountRef.current && renderer.domElement && renderer.domElement.parentNode === mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
      rendererRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once

return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
     

      <div
        ref={mountRef}
        style={{ width: "100%", maxWidth: 1550, aspectRatio: "16/9", background: "#1f2937", borderRadius: 12 }}
      />

      {!imagesLoaded ? (
        <div className="text-gray-400 mt-3">Loading site images from GCS (non-blocking)...</div>
      ) : (
        <div className="text-gray-400 mt-3"></div>
      )}
    </div>
  );
}

/* -------------------- SITE DETAIL PAGE -------------------- */
function SiteDetail() {
  const { state } = useLocation();

  if (!state) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white">
        <div className="text-center">
          <h2 className="text-2xl mb-4">Invalid site</h2>
          <a href="/" className="text-blue-400 underline">
            ← Back to Globe
          </a>
        </div>
      </div>
    );
  }

  const site = state;
  console.log("SiteDetail state:", site);        
  console.log("Image src being used:", site.image);

  return (
    <div className="flex flex-col items-center bg-gray-900 text-white min-h-screen p-8">
      <h2 className="text-3xl mb-4">{site.name}</h2>
      {site.image ? (
        <img src={site.image} alt={site.name} className="max-w-lg rounded-xl mb-6 shadow-lg" />
      ) : (
        <div className="w-full max-w-lg h-64 bg-gray-700 rounded-xl flex items-center justify-center mb-6">
          <span className="text-gray-300">No image available</span>
        </div>
      )}
      <p className="text-gray-300 max-w-2xl">{site.captions?.[0]}</p>
      <a href="/" className="text-blue-400 underline mt-8">
        ← Back to Globe
      </a>
    </div>
  );
}

/* -------------------- APP -------------------- */
export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Globe />} />
        <Route path="/site/:name" element={<SiteDetail />} />
      </Routes>
    </Router>
  );
}
