// SiteDisplay.jsx
import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import "./SiteDisplay.css";

function formatTitle(name) {
  return name.replace(/([A-Z])/g, " $1").trim();
}

// Typewriter that:
//  - types "LLAMA++ says:\n\n"
//  - pauses 2s (cursor blinking)
//  - then types the caption
function TypewriterText({ text, speed = 50 }) {
  const [displayedText, setDisplayedText] = useState("");
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    if (typeof text !== "string" || text.trim().length === 0) {
      setDisplayedText("");
      setShowCursor(false);
      return;
    }

    const prefix = "LLAMA++ says:\n\n";
    const full = prefix + text;

    let currentIndex = 0;
    let typingInterval = null;
    let pauseTimeout = null;

    setDisplayedText("");
    setShowCursor(true);

    const clearTimers = () => {
      if (typingInterval) clearInterval(typingInterval);
      if (pauseTimeout) clearTimeout(pauseTimeout);
    };

    const startTyping = () => {
      typingInterval = setInterval(() => {
        if (currentIndex >= full.length) {
          clearTimers();
          pauseTimeout = setTimeout(() => setShowCursor(false), 2000);
          return;
        }

        const nextChar = full[currentIndex];
        setDisplayedText((prev) => prev + nextChar);
        currentIndex += 1;

        // After finishing the prefix line, pause for 2 seconds
        if (currentIndex === prefix.length) {
          clearTimers();
          pauseTimeout = setTimeout(() => {
            startTyping();
          }, 2000);
        }
      }, speed);
    };

    startTyping();

    return () => {
      clearTimers();
    };
  }, [text, speed]);

  return (
    <span>
      {displayedText}
      {showCursor && <span className="typewriter-cursor">|</span>}
    </span>
  );
}

export default function SiteDisplay() {
  const { state } = useLocation();

  if (!state) {
    return (
      <div className="site-container">
        <div className="site-message">
          <h2>Invalid site</h2>
          <a href="/">
            <img
              src="/globe_back.png"
              alt="Back to Globe"
              className="globe-back-icon"
            />
          </a>
        </div>
      </div>
    );
  }

  const site = state;
  const captionText =
    Array.isArray(site.captions) && typeof site.captions[0] === "string"
      ? site.captions[0]
      : "";

  return (
    <div className="site-container">
      <h1 className="site-title">{formatTitle(site.name)}</h1>

      {site.image ? (
        <img src={site.image} alt={site.name} className="site-image" />
      ) : (
        <div className="site-image-placeholder">
          <span>No image available</span>
        </div>
      )}

      <p className="site-caption">
        <TypewriterText text={captionText} speed={40} />
      </p>

      <a href="/">
        <img
          src="/globe_back.png"
          alt="Back to Globe"
          className="globe-back-icon"
        />
      </a>
    </div>
  );
}
