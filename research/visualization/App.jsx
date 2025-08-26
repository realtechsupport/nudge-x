import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { WireframeGeometry2 } from 'three/examples/jsm/lines/WireframeGeometry2.js';
import { Wireframe } from 'three/examples/jsm/lines/Wireframe.js';


function App() {
    const mountRef = useRef(null);
    const [message, setMessage] = useState('');
    const [showModal, setShowModal] = useState(false);
    const [selectedDotData, setSelectedDotData] = useState(null); // State for clicked dot data
    const [showDotModal, setShowDotModal] = useState(false); // State to control dot modal visibility
    const [isLoadingGlobe, setIsLoadingGlobe] = useState(true); // State to manage loading indicator

    const showMessage = (msg) => {
        setMessage(msg); // Directly use setMessage as setModalContent is not defined
        setShowModal(true);
    };

    const closeModal = () => {
        setShowModal(false);
        setMessage('');
    };

    const closeDotModal = () => {
        setShowDotModal(false);
        setSelectedDotData(null);
    };

    // Function to map severity to color
    const getColorForSeverity = useCallback((severity) => {
        switch (severity) {
            case 1: return 0x00ff00; // Green (Low Severity)
            case 2: return 0xadff2f; // GreenYellow
            case 3: return 0xffff00; // Yellow (Medium Severity)
            case 4: return 0xffa500; // Orange
            case 5: return 0xff0000; // Red (High Severity)
            default: return 0x808080; // Gray for undefined severity
        }
    }, []);

    useEffect(() => {
        if (!mountRef.current) return;

        // Declare variables in a scope accessible by animate and cleanup
        let scene, camera, renderer, wireframeMaterial, controls, dotGroup, solidGlobe, wiremesh, globeGeometry, dotGeometry;

        // Scene setup
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x282c34); // Dark gray background

        // Camera setup
        camera = new THREE.PerspectiveCamera(
            75,
            mountRef.current.clientWidth / mountRef.current.clientHeight,
            0.1,
            1000
        );
        camera.position.z = 5;

        // Renderer setup
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio); // Crucial for LineMaterial and overall sharpness
        mountRef.current.appendChild(renderer.domElement);

        // Responsive canvas resizing
        const handleResize = () => {
            const width = mountRef.current.clientWidth;
            const height = mountRef.current.clientHeight;
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            // Update resolution for LineMaterial on resize
            if (wireframeMaterial) {
                wireframeMaterial.resolution.set(width * window.devicePixelRatio, height * window.devicePixelRatio);
            }
        };
        window.addEventListener('resize', handleResize);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 2); // Soft white light
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 1, 1).normalize();
        scene.add(directionalLight);

        // Globe (Sphere Geometry)
        const globeRadius = 2;
        globeGeometry = new THREE.SphereGeometry(globeRadius, 32, 32); // Assign to declared variable

        // Initial solid globe material (fallback or until texture loads)
        // Create solidGlobe immediately so it's defined for animation loop
        solidGlobe = new THREE.Mesh(globeGeometry, new THREE.MeshPhongMaterial({
            color: 0x336699, // A blue-ish color for the "earth"
            specular: 0x111111,
            shininess: 30,
        }));
        scene.add(solidGlobe); // Add the solid globe to the scene immediately

        // Load World Map Texture
        const textureLoader = new THREE.TextureLoader();
        const worldMapTextureUrl = "https://unpkg.com/three-globe@2.27.0/example/img/earth-dark.jpg"; // A more realistic map

        textureLoader.load(
            worldMapTextureUrl,
            (texture) => {
                // Update material with texture once loaded
                solidGlobe.material = new THREE.MeshBasicMaterial({
                    map: texture, // Apply the loaded texture
                });
                solidGlobe.material.needsUpdate = true; // Tell Three.js material has changed
                setIsLoadingGlobe(false); // Hide loading indicator once globe is textured
            },
            undefined, // onProgress callback
            (error) => {
                console.error('An error occurred loading the globe texture:', error);
                setMessage('Failed to load world map texture. Displaying a solid globe instead.');
                setShowModal(true); // Show the general message modal
                setIsLoadingGlobe(false); // Hide loading indicator even on error
            }
        );

        // Wiremesh material
        wireframeMaterial = new LineMaterial({ // Assign to the declared variable
            color: 0x808080, // Gray color for wiremesh
            linewidth: 0.002, // Thickness of the lines
            resolution: new THREE.Vector2(mountRef.current.clientWidth * window.devicePixelRatio, mountRef.current.clientHeight * window.devicePixelRatio),
        });

        const wireframeGeometry = new WireframeGeometry2(globeGeometry);
        wiremesh = new Wireframe(wireframeGeometry, wireframeMaterial); // Assign to declared variable
        scene.add(wiremesh); // Add the wiremesh on top of the solid globe

        // OrbitControls for interactive camera movement
        controls = new OrbitControls(camera, renderer.domElement); // Assign to declared variable
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = false;
        controls.minDistance = 2.5;
        controls.maxDistance = 10;

        // --- Simplified Dummy Data for Sites with GPS and Captions ---
        const dummySiteData = [
            { name: "Site Alpha (LA)", lat: 34.0522, lon: -118.2437, severity: 5, captions: ["The primary hazard is a large, active wildfire within a forested area. A significant smoke plume is visible, drifting eastward and impacting regional air quality."] }, // Los Angeles
            { name: "Site Beta (London)", lat: 51.5074, lon: -0.1278, severity: 4, captions: ["A significant oil spill is present in the marine environment. A dark, irregular slick, characteristic of hydrocarbons, is visible on the ocean surface."] }, // London
            { name: "Site Gamma (Sao Paulo)", lat: -23.5505, lon: -46.6333, severity: 3, captions: ["A widespread harmful algal bloom (HAB) is visible on the surface of this lake. The high concentration of algae appears as bright green and cyan swirls."] }, // São Paulo
            { name: "Site Delta (Tokyo)", lat: 35.6895, lon: 139.6917, severity: 2, captions: ["Extensive flooding is the main hazard. A river has overflowed its banks, submerging vast areas of surrounding agricultural land and settlements. "] }, // Tokyo
            { name: "Site Epsilon (New Delhi)", lat: 28.6139, lon: 77.2090, severity: 1, captions: ["A thick layer of smog and haze hangs over a major urban center. The atmospheric pollution obscures ground details and appears as a grayish-brown cloud, indicating."] }  // New Delhi
        ];

        dotGroup = new THREE.Group(); // Assign to declared variable
        const clickableObjects = []; // Array to store dots that can be intersected by raycaster

        dotGeometry = new THREE.SphereGeometry(0.05, 16, 16); // Assign to declared variable
        dummySiteData.forEach(site => {
            // Convert latitude and longitude from degrees to radians
            const latRad = site.lat * (Math.PI / 180);
            const lonRad = site.lon * (Math.PI / 180);

            // Convert spherical to Cartesian coordinates
            const x = globeRadius * Math.cos(latRad) * Math.sin(lonRad);
            const y = globeRadius * Math.sin(latRad);
            const z = globeRadius * Math.cos(latRad) * Math.cos(lonRad);

            const color = getColorForSeverity(site.severity);

            const dotMaterial = new THREE.MeshBasicMaterial({ color: color });
            const dot = new THREE.Mesh(dotGeometry, dotMaterial);
            dot.position.set(x, y, z);

            // Store the full site data in the dot's userData for easy retrieval on click
            dot.userData = site;
            dotGroup.add(dot);
            clickableObjects.push(dot); // Add to clickable array
        });
        scene.add(dotGroup);

        // Raycaster for click detection
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        const onCanvasClick = (event) => {
            // Calculate mouse position in normalized device coordinates (-1 to +1)
            // Adjust for canvas position relative to viewport
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            // Update the picking ray with the camera and mouse position
            raycaster.setFromCamera(mouse, camera);

            // Find intersected objects (dots)
            const intersects = raycaster.intersectObjects(clickableObjects);

            if (intersects.length > 0) {
                // Get the first intersected object (the closest one)
                const clickedDot = intersects[0].object;
                setSelectedDotData(clickedDot.userData); // Set the data for the modal
                setShowDotModal(true); // Show the modal
                console.log("Dot clicked:", clickedDot.userData.name); // Debugging
            } else {
                console.log("No dot clicked."); // Debugging
            }
        };

        renderer.domElement.addEventListener('click', onCanvasClick);

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);

            // Rotate the globe and dot group slightly if not controlled by user
            if (controls && !controls.isLocked) { // Check if controls are defined and not active (not being dragged)
                if (solidGlobe) solidGlobe.rotation.y += 0.001; // Ensure solidGlobe is defined
                if (wiremesh) wiremesh.rotation.y += 0.001; // Ensure wiremesh is defined
                if (dotGroup) dotGroup.rotation.y += 0.001; // Ensure dotGroup is defined
            }

            if (controls) controls.update(); // Ensure controls are defined
            if (renderer && scene && camera) renderer.render(scene, camera); // Ensure renderer, scene, camera are defined
        };

        animate(); // Start the animation loop

        // Cleanup function
        return () => {
            window.removeEventListener('resize', handleResize);
            if (renderer && renderer.domElement) { // Check if renderer and its domElement exist before removing listener
                renderer.domElement.removeEventListener('click', onCanvasClick); // Remove click listener
                if (mountRef.current && renderer.domElement.parentNode === mountRef.current) { // Check parent before removing
                    mountRef.current.removeChild(renderer.domElement);
                }
            }
            // Dispose Three.js resources
            if (renderer) renderer.dispose();
            if (wireframeMaterial) wireframeMaterial.dispose();
            // solidGlobeMaterial is disposed inside texture loader callback or fallback, or here if it's the initial one
            if (solidGlobe && solidGlobe.material) {
                solidGlobe.material.dispose();
            }
            if (globeGeometry) globeGeometry.dispose();
            if (dotGeometry) dotGeometry.dispose();
            if (dotGroup) {
                dotGroup.children.forEach(dot => {
                    if (dot.material) dot.material.dispose();
                });
                dotGroup.clear();
                scene.remove(dotGroup);
            }
            if (solidGlobe) scene.remove(solidGlobe);
            if (wiremesh) scene.remove(wiremesh);
            if (controls) controls.dispose();
        };
    }, [getColorForSeverity, showMessage]);

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white font-inter p-4">
            <h1 className="text-4xl font-bold mb-6 text-center">MLLM Response Severity Globe</h1>
            <p className="text-lg text-gray-300 mb-8 text-center max-w-2xl">
                This interactive 3D globe visualizes generated sites, color-coded by the severity of MLLM responses.
                <br />
                <span className="font-semibold">Drag the globe with your mouse to rotate it! Click a dot for details.</span>
            </p>

            <div
                ref={mountRef}
                className="w-full max-w-4xl aspect-video bg-gray-800 rounded-xl shadow-2xl overflow-hidden relative"
                style={{ minHeight: '400px' }}
            >
                {isLoadingGlobe && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 z-10">
                        <div className="text-white text-xl">Loading Globe...</div>
                    </div>
                )}
                {/* 3D scene will be rendered here */}
            </div>

            <div className="mt-8 p-4 bg-gray-800 rounded-lg shadow-lg">
                <h2 className="text-2xl font-semibold mb-4 text-center">Severity Color Key</h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                    <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 rounded-full" style={{ backgroundColor: '#00ff00' }}></div>
                        <span>1 (Low)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 rounded-full" style={{ backgroundColor: '#adff2f' }}></div>
                        <span>2 (GreenYellow)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 rounded-full" style={{ backgroundColor: '#ffff00' }}></div>
                        <span>3 (Yellow)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 rounded-full" style={{ backgroundColor: '#ffa500' }}></div>
                        <span>4 (Orange)</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 rounded-full" style={{ backgroundColor: '#ff0000' }}></div>
                        <span>5 (Red)</span>
                    </div>
                </div>
            </div>

            {/* General Message Modal */}
            {showModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white p-6 rounded-lg shadow-xl text-center max-w-sm">
                        <p className="text-gray-800 text-lg mb-4">{message}</p>
                        <button
                            onClick={closeModal}
                            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-300"
                        >
                            OK
                        </button>
                    </div>
                </div>
            )}

            {/* Dot Details Modal */}
            {showDotModal && selectedDotData && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white p-6 rounded-lg shadow-xl text-gray-800 max-w-md w-full mx-4">
                        <h3 className="text-2xl font-bold mb-3 text-blue-700">{selectedDotData.name}</h3>
                        <p className="mb-2">
                            <span className="font-semibold">Location:</span> Lat {selectedDotData.lat.toFixed(4)}, Lon {selectedDotData.lon.toFixed(4)}
                        </p>
                        <p className="mb-4">
                            <span className="font-semibold">Severity:</span> {selectedDotData.severity}
                            <span
                                className="inline-block w-4 h-4 rounded-full ml-2 align-middle"
                                style={{ backgroundColor: `#${getColorForSeverity(selectedDotData.severity).toString(16).padStart(6, '0')}` }}
                            ></span>
                        </p>
                        <h4 className="text-xl font-semibold mb-2 text-gray-700">MLLM Captions:</h4>
                        <ul className="list-disc list-inside text-left mb-4">
                            {selectedDotData.captions.map((caption, index) => (
                                <li key={index} className="mb-1">{caption}</li>
                            ))}
                        </ul>
                        <button
                            onClick={closeDotModal}
                            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-300"
                        >
                            Close
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default App;
