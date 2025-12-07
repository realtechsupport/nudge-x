import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import "./SiteDisplay.css";

function formatTitle(name) {
  return name.replace(/([A-Z])/g, " $1").trim();
}

function TypewriterText({ text, speed = 50 }) {
  const [displayedText, setDisplayedText] = useState("");
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    // Guard against non-string or empty input
    if (typeof text !== "string" || text.trim().length === 0) {
      setDisplayedText("");
      setShowCursor(false);
      return;
    }

    let currentIndex = 0;
    const chars = [...text];

    // Reset when the text prop changes
    setDisplayedText("");
    setShowCursor(true);

    const typingInterval = setInterval(() => {
      if (currentIndex < chars.length) {
        const nextChar = chars[currentIndex];
        setDisplayedText((prev) => prev + nextChar);
        currentIndex += 1;

        if (currentIndex === chars.length) {
          clearInterval(typingInterval);
          setTimeout(() => setShowCursor(false), 2000);
        }
      } else {
        // Extra safety: never read past the end
        clearInterval(typingInterval);
      }
    }, speed);

    return () => clearInterval(typingInterval);
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
