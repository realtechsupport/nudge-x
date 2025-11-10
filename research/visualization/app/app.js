import React, { useRef, useEffect, useState, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams } from 'react-router-dom';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { WireframeGeometry2 } from 'three/examples/jsm/lines/WireframeGeometry2.js';
import { Wireframe } from 'three/examples/jsm/lines/Wireframe.js';

// -------------------- SITE DATA --------------------
const siteData = [
  { name: "Site Alpha (LA)", lat: 34.0522, lon: -118.2437, image: 'http://35.208.244.178/images/LosAngeles.png', severity: 5, captions: ["The primary hazard is a large, active wildfire within a forested area. A significant smoke plume is visible, drifting eastward and impacting regional air quality."] },
  { name: "Site Beta (London)", lat: 51.5074, lon: -0.1278, image: 'http://35.208.244.178/images/London.png', severity: 4, captions: ["A significant oil spill is present in the marine environment."] },
  { name: "Site Gamma (Islamabad)", lat: 33.6996, lon: 73.0362, image: 'http://35.208.244.178/images/Islamabad.png', severity: 3, captions: ["This satellite image shows Islamabad with its planned grid-like urban sectors."] },
  { name: "Site Delta (Paris)", lat: 48.8575, lon: 2.3514, image: 'http://35.208.244.178/images/Paris.png', severity: 2, captions: ["This satellite image shows Paris with the Seine River running through the center."] },
  { name: "Site Epsilon (Varanasi)", lat: 25.316, lon: 82.9739, image: 'http://35.208.244.178/images/Varanasi.png', severity: 1, captions: ["This January 2025 satellite image shows Varanasi along the Ganga River."] }
];

// -------------------- GLOBE COMPONENT --------------------
function Globe() {
  const mountRef = useRef(null);
  const [isLoadingGlobe, setIsLoadingGlobe] = useState(true);
  const navigate = useNavigate();

  const getColorForSeverity = useCallback((severity) => {
    switch (severity) {
      case 1: return 0xffa500;
      default: return 0xffa500;
    }
  }, []);

  useEffect(() => {
    if (!mountRef.current) return;

    let scene, camera, renderer, wireframeMaterial, controls, dotGroup, solidGlobe, wiremesh, globeGeometry, dotGeometry;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x282c34);

    camera = new THREE.PerspectiveCamera(75, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
    camera.position.z = 5;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    
    const handleResize = () => {
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      if (wireframeMaterial) {
        wireframeMaterial.resolution.set(width * window.devicePixelRatio, height * window.devicePixelRatio);
      }
    };
    window.addEventListener('resize', handleResize);

    scene.add(new THREE.AmbientLight(0x404040, 2));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    const globeRadius = 2;
    globeGeometry = new THREE.SphereGeometry(globeRadius, 32, 32);
    solidGlobe = new THREE.Mesh(globeGeometry, new THREE.MeshPhongMaterial({ color: 0x336699, specular: 0x111111, shininess: 30 }));
    scene.add(solidGlobe);

    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(
      "https://unpkg.com/three-globe@2.27.0/example/img/earth-dark.jpg",
      (texture) => {
        solidGlobe.material = new THREE.MeshBasicMaterial({ map: texture });
        solidGlobe.material.needsUpdate = true;
        setIsLoadingGlobe(false);
      },
      undefined,
      (err) => {
        console.error("Error loading globe texture:", err);
        setIsLoadingGlobe(false);
      }
    );

    wireframeMaterial = new LineMaterial({
      color: 0x808080,
      linewidth: 0.002,
      resolution: new THREE.Vector2(mountRef.current.clientWidth * window.devicePixelRatio, mountRef.current.clientHeight * window.devicePixelRatio)
    });
    wiremesh = new Wireframe(new WireframeGeometry2(globeGeometry), wireframeMaterial);
    scene.add(wiremesh);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 2.5;
    controls.maxDistance = 10;

    dotGroup = new THREE.Group();
    dotGeometry = new THREE.SphereGeometry(0.02, 16, 16);
    const clickableObjects = [];

    siteData.forEach(site => {
      const latRad = site.lat * (Math.PI / 180);
      const lonRad = -site.lon * (Math.PI / 180);
      const x = globeRadius * Math.cos(latRad) * Math.cos(lonRad);
      const y = globeRadius * Math.sin(latRad);
      const z = globeRadius * Math.cos(latRad) * Math.sin(lonRad);

      const dot = new THREE.Mesh(dotGeometry, new THREE.MeshBasicMaterial({ color: getColorForSeverity(site.severity) }));
      dot.position.set(x, y, z);
      dot.userData = site;

      dotGroup.add(dot);
      clickableObjects.push(dot);
    });

    scene.add(dotGroup);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onCanvasClick = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(clickableObjects);

      if (intersects.length > 0) {
        const site = intersects[0].object.userData;
        // Navigate to detail page with state
        navigate(`/site/${encodeURIComponent(site.name)}`, { state: site });
      }
    };

    renderer.domElement.addEventListener('click', onCanvasClick);

    const animate = () => {
      requestAnimationFrame(animate);
      solidGlobe.rotation.y += 0.001;
      wiremesh.rotation.y += 0.001;
      dotGroup.rotation.y += 0.001;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('click', onCanvasClick);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }

      renderer.dispose();
      wireframeMaterial.dispose();
      solidGlobe.material.dispose();
      globeGeometry.dispose();
      dotGeometry.dispose();
      dotGroup.children.forEach(dot => dot.material.dispose());
      dotGroup.clear();
      scene.remove(dotGroup);
      scene.remove(solidGlobe);
      scene.remove(wiremesh);
      controls.dispose();
    };
  }, [getColorForSeverity, navigate]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white font-inter p-4">
      <h1 className="text-4xl font-bold mb-6 text-center">Mining Sites Globe</h1>
      <p className="text-lg text-gray-300 mb-8 text-center max-w-2xl">
        Click on a mining site to view its satellite image and description.
      </p>

      <div
        ref={mountRef}
        className="w-full max-w-4xl aspect-video bg-gray-800 rounded-xl shadow-2xl overflow-hidden relative"
        style={{ minHeight: '635px' }}
      >
        {isLoadingGlobe && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 z-10">
            <div className="text-white text-xl">Loading Globe...</div>
          </div>
        )}
      </div>
    </div>
  );
}

// -------------------- SITE DETAIL PAGE --------------------
function SiteDetail() {
  const { name } = useParams();
  const site = siteData.find(s => s.name === decodeURIComponent(name));
  if (!site) return <div className="text-white text-center p-8">Site not found.</div>;

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white font-inter p-4">
      <h2 className="text-3xl font-bold mb-4">{site.name}</h2>
      <img src={site.image} alt={site.name} className="max-w-lg rounded-xl mb-6 border border-gray-700" />
      <p className="text-gray-300 max-w-2xl text-center">{site.captions[0]}</p>
      <a href="/" className="mt-8 text-blue-400 underline hover:text-blue-300">← Back to Globe</a>
    </div>
  );
}

// -------------------- MAIN APP --------------------
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
