// SiteDisplay.jsx
import React, { useState, useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";
import "./SiteDisplay.css";

function formatLabel(label) {
  if (typeof label !== "string") return "";
  let result = label.trim();
  if (!result) return "";

  // Only split CamelCase if there are no spaces yet (e.g. "AmableMine")
  if (!/\s/.test(result)) {
    result = result.replace(/([A-Z])/g, " $1").trim();
  }

  // Collapse any double spaces
  return result.replace(/\s+/g, " ");
}

function TypewriterText({ text, speed = 50, onDone }) {
  const [displayedText, setDisplayedText] = useState("");
  const [showCursor, setShowCursor] = useState(true);
  const hasCalledDoneRef = useRef(false);

  useEffect(() => {
    // Guard: empty / non‑string text
    if (typeof text !== "string" || text.trim().length === 0) {
      setDisplayedText("");
      setShowCursor(false);
      return;
    }

    let currentIndex = 0;
    const chars = [...text];

    // Reset whenever the *text* changes
    setDisplayedText("");
    setShowCursor(true);
    hasCalledDoneRef.current = false;

    let typingInterval;

    const startTyping = () => {
      typingInterval = setInterval(() => {
        if (currentIndex < chars.length) {
          const nextChar = chars[currentIndex];
          setDisplayedText((prev) => prev + nextChar);
          currentIndex += 1;

          if (currentIndex === chars.length) {
            clearInterval(typingInterval);
            // Let the cursor blink ~2s, then hide it,
            // THEN wait 1s before firing onDone once ( → globe icon )
            setTimeout(() => {
              setShowCursor(false);
              setTimeout(() => {
                if (onDone && !hasCalledDoneRef.current) {
                  hasCalledDoneRef.current = true;
                  onDone();
                }
              }, 1000); // 1s after cursor stops blinking
            }, 2000); // ~2s cursor blink
          }
        } else {
          clearInterval(typingInterval);
        }
      }, speed);
    };

    // Wait 1s before starting the caption typing
    const delayTimeout = setTimeout(startTyping, 1000);

    return () => {
      clearTimeout(delayTimeout);
      if (typingInterval) {
        clearInterval(typingInterval);
      }
    };
  }, [text, speed]); // keep onDone OUT to avoid re‑running & double typing

  return (
    <span>
      {displayedText}
      {showCursor && <span className="typewriter-cursor" />}
    </span>
  );
}

export default function SiteDisplay() {
  const { state } = useLocation();

  // Fallback: direct URL / bad navigation
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

  const captionBody =
    Array.isArray(site.captions) && typeof site.captions[0] === "string"
      ? site.captions[0]
      : "";

  // Prefix + blank line + caption
  const fullCaptionText = `Llama4++ reports:\n\n${captionBody}`;

  const [showBackIcon, setShowBackIcon] = useState(false);
  
  // --- NEW: mine, location, country title ---
  const mineTitle = formatLabel(site.mine);
  const locationTitle = formatLabel(site.name);
  const countryTitle = formatLabel(site.country);

  let fullTitle = "";

  if (mineTitle && locationTitle && countryTitle) {
    fullTitle = `${mineTitle}, ${locationTitle}, ${countryTitle}`;
  } else if (mineTitle && locationTitle) {
    fullTitle = `${mineTitle}, ${locationTitle}`;
  } else if (mineTitle && countryTitle) {
    fullTitle = `${mineTitle}, ${countryTitle}`;
  } else if (locationTitle && countryTitle) {
    fullTitle = `${locationTitle}, ${countryTitle}`;
  } else {
    fullTitle = mineTitle || locationTitle || countryTitle || "";
  }

  return (
    <div className="site-container">
      <h1 className="site-title">{fullTitle}</h1>

      <div className="site-content-row">
        {site.image ? (
          <img src={site.image} alt={site.name} className="site-image" />
        ) : (
          <div className="site-image-placeholder">
            <span>No image available</span>
          </div>
        )}

        <div className="site-caption-block">
          <TypewriterText
            text={fullCaptionText}
            speed={40}
            onDone={() => setShowBackIcon(true)}
          />
        </div>
      </div>

      {showBackIcon && (
        <a href="/" className="site-link">
          <img
            src="/globe_back.png"
            alt="Back to Globe"
            className="globe-back-icon"
          />
        </a>
      )}
    </div>
  );
}



