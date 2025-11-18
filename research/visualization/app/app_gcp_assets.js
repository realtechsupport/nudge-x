import React, { useRef, useEffect, useState, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useParams } from 'react-router-dom';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { WireframeGeometry2 } from 'three/examples/jsm/lines/WireframeGeometry2.js';
import { Wireframe } from 'three/examples/jsm/lines/Wireframe.js';

// Define the base URL for the GCP storage bucket (assuming public/shared access)
const GCP_BUCKET_BASE_URL = 'https://storage.googleapis.com/nudge-bucket/';

// -------------------- SITE DATA --------------------
// Reverting to direct 'image' property with full public URL
const siteData = [
  { name: "Site Alpha (LA)", lat: 34.0522, lon: -118.2437, image: `${GCP_BUCKET_BASE_URL}LosAngeles.png`, severity: 5, captions: ["The primary hazard is a large, active wildfire within a forested area. A significant smoke plume is visible, drifting eastward and impacting regional air quality."] },
  { name: "Site Beta (London)", lat: 51.5074, lon: -0.1278, image: `${GCP_BUCKET_BASE_URL}London.png`, severity: 4, captions: ["A significant oil spill is present in the marine environment."] },
  { name: "Site Gamma (Islamabad)", lat: 33.6996, lon: 73.0362, image: `${GCP_BUCKET_BASE_URL}Islamabad.png`, severity: 3, captions: ["This satellite image shows Islamabad with its planned grid-like urban sectors."] },
  { name: "Site Delta (Paris)", lat: 48.8575, lon: 2.3514, image: `${GCP_BUCKET_BASE_URL}Paris.png`, severity: 2, captions: ["This satellite image shows Paris with the Seine River running through the center."] },
  { name: "Site Epsilon (Varanasi)", lat: 25.316, lon: 82.9739, image: `${GCP_BUCKET_BASE_URL}Varanasi.png`, severity: 1, captions: ["This January 2025 satellite image shows Varanasi along the Ganga River."] }
];

// -------------------- GLOBE COMPONENT --------------------
function Globe() {
  const mountRef = useRef(null);
  const [isLoadingGlobe, setIsLoadingGlobe] = useState(true);
  const navigate = useNavigate();

  const getColorForSeverity = useCallback((severity) => {
    switch (severity) {
      case 5: return 0xff0000; // High Severity: Red
      case 4: return 0xff4500; // Medium-High Severity: Orange-Red
      case 3: return 0xffa500; // Medium Severity: Orange
      case 2: return 0xffff00; // Low-Medium Severity: Yellow
      case 1: return 0x00ff00; // Low Severity: Green
      default: return 0xffa500;
    }
  }, []);

  useEffect(() => {
    if (!mountRef.current) return;

    let scene, camera, renderer, wireframeMaterial, controls, dotGroup, solidGlobe, wiremesh, globeGeometry, dotGeometry;

    // --- Setup Scene, Camera, Renderer ---
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x282c34);

    camera = new THREE.PerspectiveCamera(75, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
    camera.position.z = 5;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    
    // --- Resize Handler ---
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

    // --- Lighting ---
    scene.add(new THREE.AmbientLight(0x404040, 2));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    // --- Globe Mesh (Solid) ---
    const globeRadius = 2;
    globeGeometry = new THREE.SphereGeometry(globeRadius, 32, 32);
    solidGlobe = new THREE.Mesh(globeGeometry, new THREE.MeshPhongMaterial({ color: 0x336699, specular: 0x111111, shininess: 30 }));
    scene.add(solidGlobe);

    // Load Earth Texture
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

    // --- Wireframe Mesh ---
    wireframeMaterial = new LineMaterial({
      color: 0x808080,
      linewidth: 0.002,
      resolution: new THREE.Vector2(mountRef.current.clientWidth * window.devicePixelRatio, mountRef.current.clientHeight * window.devicePixelRatio)
    });
    wiremesh = new Wireframe(new WireframeGeometry2(globeGeometry), wireframeMaterial);
    scene.add(wiremesh);

    // --- Orbit Controls ---
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 2.5;
    controls.maxDistance = 10;

    // --- Data Points (Dots) ---
    dotGroup = new THREE.Group();
    dotGeometry = new THREE.SphereGeometry(0.02, 16, 16);
    const clickableObjects = [];

    siteData.forEach(site => {
      // Convert lat/lon to 3D spherical coordinates
      const latRad = site.lat * (Math.PI / 180);
      const lonRad = -site.lon * (Math.PI / 180);
      const x = globeRadius * Math.cos(latRad) * Math.cos(lonRad);
      const y = globeRadius * Math.sin(latRad);
      const z = globeRadius * Math.cos(latRad) * Math.sin(lonRad);

      const dot = new THREE.Mesh(dotGeometry, new THREE.MeshBasicMaterial({ color: getColorForSeverity(site.severity) }));
      dot.position.set(x, y, z);
      dot.userData = site; // Attach site data to the dot

      dotGroup.add(dot);
      clickableObjects.push(dot);
    });

    scene.add(dotGroup);

    // --- Interaction (Raycasting) ---
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
        // Navigate to detail page
        navigate(`/site/${encodeURIComponent(site.name)}`, { state: site });
      }
    };

    renderer.domElement.addEventListener('click', onCanvasClick);

    // --- Animation Loop ---
    const animate = () => {
      requestAnimationFrame(animate);
      // Continuous slow rotation for the entire group
      solidGlobe.rotation.y += 0.001;
      wiremesh.rotation.y += 0.001;
      dotGroup.rotation.y += 0.001;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // --- Cleanup Function ---
    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('click', onCanvasClick);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }

      // Dispose of Three.js objects to free memory
      renderer.dispose();
      if (wireframeMaterial) wireframeMaterial.dispose();
      if (solidGlobe && solidGlobe.material) solidGlobe.material.dispose();
      if (globeGeometry) globeGeometry.dispose();
      if (dotGeometry) dotGeometry.dispose();
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
  
  // Note: Since we are using a direct public URL now, the state management 
  // for signed URLs (signedImageUrl, isLoadingImage, imageError, fetchSignedUrl) 
  // is removed to simplify the component.

  if (!site) return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white font-inter p-4">
      <h2 className="text-3xl font-bold mb-4">Site Not Found</h2>
      <p className="text-gray-300">The requested site could not be located in the data.</p>
      <a href="/" className="mt-8 text-blue-400 underline hover:text-blue-300">← Back to Globe</a>
    </div>
  );

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white font-inter p-4">
      <h2 className="text-3xl font-bold mb-4 text-center">{site.name}</h2>
      
      <div className="max-w-full md:max-w-lg rounded-xl mb-6 border border-gray-700 shadow-xl overflow-hidden">
        <img 
          src={site.image} 
          alt={`Satellite image of ${site.name}`} 
          className="w-full h-auto"
          // Fallback in case the image cannot be loaded from the public bucket
          onError={(e) => { 
              e.target.onerror = null; 
              e.target.src="https://placehold.co/500x300/374151/ffffff?text=Image+Unavailable"; 
          }}
        />
      </div>

      <p className="text-gray-300 max-w-2xl text-center text-lg">{site.captions[0]}</p>
      
      <div className={`mt-4 p-2 rounded-lg font-mono text-sm ${site.severity > 3 ? 'bg-red-900 text-red-300' : 'bg-green-900 text-green-300'}`}>
        Severity: {site.severity}
      </div>

      <a href="/" className="mt-8 text-blue-400 underline hover:text-blue-300 transition duration-150 ease-in-out">← Back to Globe</a>
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