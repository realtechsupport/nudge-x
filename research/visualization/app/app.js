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
    const [selectedDotData, setSelectedDotData] = useState(null); 
    const [showDotModal, setShowDotModal] = useState(false); 
    const [isLoadingGlobe, setIsLoadingGlobe] = useState(true); 
    
    // --- Gemini API Feature States ---
    const [geminiAnalysis, setGeminiAnalysis] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    // --- End Gemini API Feature States ---


    const showMessage = (msg) => {
        setMessage(msg);
        setShowModal(true);
    };

    const closeModal = () => {
        setShowModal(false);
        setMessage('');
    };

    const closeDotModal = () => {
        setShowDotModal(false);
        setSelectedDotData(null);
        setGeminiAnalysis(null); // Clear analysis when closing modal
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

    // --- Gemini API Call Function ---
    const handleGenerateAnalysis = useCallback(async () => {
        if (!selectedDotData || isAnalyzing) return;

        setIsAnalyzing(true);
        setGeminiAnalysis("Analyzing data...");

        try {
            const prompt = `Analyze the following disaster site data: Location: ${selectedDotData.name} (${selectedDotData.lat.toFixed(2)}, ${selectedDotData.lon.toFixed(2)}). Severity Score: ${selectedDotData.severity}/5. Raw MLLM Captions: ${selectedDotData.captions.join('; ')}. Based on this, provide a concise summary of the primary risk and suggest one immediate next step for response teams. Your response should use Markdown.`;
            
            const systemPrompt = "You are a Geospatial Risk Analyst providing actionable insights. Your response must be in Markdown format with two clear headings: 'Primary Risk Summary' and 'Immediate Action'.";

            const apiKey = ""; 
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;

            const payload = {
                contents: [{ parts: [{ text: prompt }] }],
                systemInstruction: { parts: [{ text: systemPrompt }] },
            };

            const fetchConfig = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            };

            // Simple retry mechanism (up to 3 times)
            let response;
            for (let i = 0; i < 3; i++) {
                response = await fetch(apiUrl, fetchConfig);
                if (response.ok) break;
                console.warn(`Attempt ${i + 1} failed. Retrying in ${2 ** i}s...`);
                await new Promise(resolve => setTimeout(resolve, 1000 * (2 ** i)));
            }
            
            if (!response.ok) {
                throw new Error(`API request failed with status: ${response.status}`);
            }

            const result = await response.json();
            const text = result.candidates?.[0]?.content?.parts?.[0]?.text || "Analysis failed to generate text.";
            setGeminiAnalysis(text);

        } catch (error) {
            console.error("Gemini API Error:", error);
            setGeminiAnalysis("Error generating analysis. Please check console.");
        } finally {
            setIsAnalyzing(false);
        }
    }, [selectedDotData, isAnalyzing]); // Depend on selectedDotData to ensure we analyze the right data

    // --- End Gemini API Call Function ---

    useEffect(() => {
        if (!mountRef.current) return;

        // Ensure all objects that need to be accessed in `animate` are declared here
        let scene, camera, renderer, wireframeMaterial, controls, dotGroup, solidGlobe, wiremesh, globeGeometry, dotGeometry;

        // Scene setup
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x282c34);

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
        renderer.setPixelRatio(window.devicePixelRatio);
        mountRef.current.appendChild(renderer.domElement);

        // Responsive canvas resizing
        const handleResize = () => {
            const width = mountRef.current.clientWidth;
            const height = mountRef.current.clientHeight;
            if (renderer && camera) {
                renderer.setSize(width, height);
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
            }
            if (wireframeMaterial) {
                wireframeMaterial.resolution.set(width * window.devicePixelRatio, height * window.devicePixelRatio);
            }
        };
        window.addEventListener('resize', handleResize);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 2);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 1, 1).normalize();
        scene.add(directionalLight);

        // Globe (Sphere Geometry)
        const globeRadius = 2;
        globeGeometry = new THREE.SphereGeometry(globeRadius, 32, 32);

        // Initial solid globe material (Fallback/Initial state)
        solidGlobe = new THREE.Mesh(globeGeometry, new THREE.MeshPhongMaterial({
            color: 0x336699,
            specular: 0x111111,
            shininess: 30,
        }));
        scene.add(solidGlobe); // Add solidGlobe to scene immediately

        // Load World Map Texture
        const textureLoader = new THREE.TextureLoader();
        const worldMapTextureUrl = "https://unpkg.com/three-globe@2.27.0/example/img/earth-dark.jpg"; 

        textureLoader.load(
            worldMapTextureUrl,
            (texture) => {
                solidGlobe.material = new THREE.MeshBasicMaterial({
                    map: texture,
                });
                solidGlobe.material.needsUpdate = true;
                setIsLoadingGlobe(false);
            },
            undefined, 
            (error) => {
                console.error('An error occurred loading the globe texture:', error);
                setMessage('Failed to load world map texture. Displaying a solid globe instead.');
                setShowModal(true);
                setIsLoadingGlobe(false);
            }
        );

        // Wiremesh material
        wireframeMaterial = new LineMaterial({
            color: 0x808080,
            linewidth: 0.002,
            resolution: new THREE.Vector2(mountRef.current.clientWidth * window.devicePixelRatio, mountRef.current.clientHeight * window.devicePixelRatio),
        });

        const wireframeGeometry = new WireframeGeometry2(globeGeometry);
        wiremesh = new Wireframe(wireframeGeometry, wireframeMaterial);
        scene.add(wiremesh);

        // OrbitControls for interactive camera movement
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = false;
        controls.minDistance = 2.5;
        controls.maxDistance = 10;

        // --- Simplified Dummy Data for Sites with GPS and Captions ---
        const dummySiteData = [
            { name: "Site Alpha (LA)", lat: 34.0522, lon: -118.2437, severity: 5, captions: ["The primary hazard is a large, active wildfire within a forested area. A significant smoke plume is visible, drifting eastward and impacting regional air quality."] },
            { name: "Site Beta (London)", lat: 51.5074, lon: -0.1278, severity: 4, captions: ["A significant oil spill is present in the marine environment. A dark, irregular slick, characteristic of hydrocarbons, is visible on the ocean surface."] },
            { name: "Site Gamma (Sao Paulo)", lat: -23.5505, lon: -46.6333, severity: 3, captions: ["A widespread harmful algal bloom (HAB) is visible on the surface of this lake. The high concentration of algae appears as bright green and cyan swirls."] },
            { name: "Site Delta (Tokyo)", lat: 35.6895, lon: 139.6917, severity: 2, captions: ["Extensive flooding is the main hazard. A river has overflowed its banks, submerging vast areas of surrounding agricultural land and settlements. "] },
            { name: "Site Epsilon (New Delhi)", lat: 28.6139, lon: 77.2090, severity: 1, captions: ["A thick layer of smog and haze hangs over a major urban center. The atmospheric pollution obscures ground details and appears as a grayish-brown cloud, indicating."] }
        ];

        dotGroup = new THREE.Group();
        const clickableObjects = [];

        dotGeometry = new THREE.SphereGeometry(0.05, 16, 16);
        dummySiteData.forEach(site => {
            const latRad = site.lat * (Math.PI / 180);
            const lonRad = site.lon * (Math.PI / 180);

            const x = globeRadius * Math.cos(latRad) * Math.sin(lonRad);
            const y = globeRadius * Math.sin(latRad);
            const z = globeRadius * Math.cos(latRad) * Math.cos(lonRad);

            const color = getColorForSeverity(site.severity);

            const dotMaterial = new THREE.MeshBasicMaterial({ color: color });
            const dot = new THREE.Mesh(dotGeometry, dotMaterial);
            dot.position.set(x, y, z);

            dot.userData = site;
            dotGroup.add(dot);
            clickableObjects.push(dot);
        });
        scene.add(dotGroup);

        // Raycaster for click detection
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        const onCanvasClick = (event) => {
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(clickableObjects);

            if (intersects.length > 0) {
                const clickedDot = intersects[0].object;
                setSelectedDotData(clickedDot.userData);
                setShowDotModal(true);
                setGeminiAnalysis(null); // Reset analysis when a new dot is clicked
                console.log("Dot clicked:", clickedDot.userData.name);
            } else {
                console.log("No dot clicked.");
            }
        };

        renderer.domElement.addEventListener('click', onCanvasClick);

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);

            // Accessing solidGlobe, wiremesh, and dotGroup is now safe because they are defined above
            if (controls && !controls.isLocked) {
                solidGlobe.rotation.y += 0.001; 
                wiremesh.rotation.y += 0.001;
                dotGroup.rotation.y += 0.001;
            }

            if (controls) controls.update();
            if (renderer && scene && camera) renderer.render(scene, camera);
        };

        animate();

        // Cleanup function
        return () => {
            window.removeEventListener('resize', handleResize);
            if (renderer && renderer.domElement) {
                renderer.domElement.removeEventListener('click', onCanvasClick);
                if (mountRef.current && renderer.domElement.parentNode === mountRef.current) {
                    mountRef.current.removeChild(renderer.domElement);
                }
            }
            // Dispose Three.js resources
            if (renderer) renderer.dispose();
            if (wireframeMaterial) wireframeMaterial.dispose();
            // Dispose geometries and materials
            if (solidGlobe && solidGlobe.material) solidGlobe.material.dispose();
            if (globeGeometry) globeGeometry.dispose();
            if (dotGeometry) dotGeometry.dispose();
            
            // Clean up group and scene elements
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
    }, [getColorForSeverity, showMessage, handleGenerateAnalysis]);

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
                    <div className="bg-white p-6 rounded-lg shadow-xl text-gray-800 max-w-xl w-full mx-4 overflow-y-auto max-h-[90vh]">
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
                        
                        <h4 className="text-xl font-semibold mb-2 text-gray-700 border-b pb-1">Raw MLLM Captions:</h4>
                        <ul className="list-disc list-inside text-left text-sm mb-4">
                            {selectedDotData.captions.map((caption, index) => (
                                <li key={index} className="mb-1">{caption}</li>
                            ))}
                        </ul>

                        <div className="border p-3 rounded-lg bg-gray-50 mt-4">
                            <h4 className="text-xl font-semibold mb-2 text-purple-700">Gemini Risk Analysis ✨</h4>
                            
                            {!geminiAnalysis ? (
                                <button
                                    onClick={handleGenerateAnalysis}
                                    disabled={isAnalyzing}
                                    className={`px-4 py-2 w-full rounded-md text-white transition duration-300 ${isAnalyzing ? 'bg-purple-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700'}`}
                                >
                                    {isAnalyzing ? (
                                        <>
                                            <span className="animate-spin inline-block mr-2">⚙️</span>
                                            Analyzing...
                                        </>
                                    ) : (
                                        "Generate Risk Analysis ✨"
                                    )}
                                </button>
                            ) : (
                                <div className="text-left text-sm whitespace-pre-wrap">
                                    {/* Display the Markdown content directly */}
                                    <pre className="p-2 bg-white border rounded-md text-sm whitespace-pre-wrap">
                                        {geminiAnalysis}
                                    </pre>
                                </div>
                            )}
                        </div>
                        
                        <button
                            onClick={closeDotModal}
                            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-300 mt-6"
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
