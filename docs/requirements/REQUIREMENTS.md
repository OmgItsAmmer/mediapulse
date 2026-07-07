# Final Requirements Comparison

This document provides a comparative analysis of the client requirements for the Media Monitoring systems across three clients: **DGPR**, **FOCUS 360**, and **SINDH**.

The comparison is divided into three sections:
1. **Overlapping Features (Core Platform features common to all clients)**
2. **Partially Overlapping Features (Features with shared goals but different scope, scale, or implementation details)**
3. **Non-Overlapping Features (Unique requirements specific to individual clients)**

---

## 1. Overlapping Features
*These core features are required by all three clients and represent the foundational pillars of the media monitoring platform.*

### A. Role-Based Access Control (RBAC) & User Security
*   **DGPR:** Role-Based Access Control (RBAC), Single Sign-On (SSO), and comprehensive audit trails.
*   **FOCUS 360:** RBAC, Multi-Factor Authentication (MFA), SSO integration, and activity logs.
*   **SINDH:** RBAC, account management, and detailed activity logs.

### B. Natural Language Processing (NLP) & Sentiment Analysis
*   **DGPR:** Sentiment analysis, Named Entity Recognition (NER), topic modeling, and language identification.
*   **FOCUS 360:** AI-powered NLP sentiment analysis (positive, negative, neutral), keyword, and topic tagging.
*   **SINDH:** AI-driven sentiment classification of text, audio, and video content (positive, negative, neutral), and NER (extracting people, places, organizations).

### C. Speech-to-Text (Transcription)
*   **DGPR:** Speech-to-Text conversion supporting Urdu and English.
*   **FOCUS 360:** AI-powered transcription of live or recorded electronic media broadcasts (multilingual).
*   **SINDH:** Multi-lingual speech-to-text supporting Urdu, English, Pashto, and Sindhi.

### D. Multi-Platform Ingestion (Social Media & Web Crawling)
*   **DGPR:** Social media ingestion (Facebook, X/Twitter, Instagram, TikTok, YouTube) via official APIs and isolated sandboxed crawlers.
*   **FOCUS 360:** Data collection using web crawlers (Scrapy, BeautifulSoup, Selenium) with proxy rotation.
*   **SINDH:** Real-time social media tracking including 200 Twitter handles, 100 YouTube channels, and 100 web pages.

### E. Search Engine & Analytics Dashboards
*   **DGPR:** Elasticsearch-powered multilingual search, web-based MAM console, and custom analytics dashboards.
*   **FOCUS 360:** Relational database/Elasticsearch query engine, real-time web dashboard with customizable charts and heatmaps.
*   **SINDH:** Dynamic search engine (text, speaker, and image queries) and analytics dashboard with summary graphs and heatmaps.

### F. Automated Reporting & Archival
*   **DGPR:** Tiered archival to S3-compatible storage, custom workflow reports.
*   **FOCUS 360:** Relational/data lake storage, automated reports (Engagement, Sentiment, Trend, Crisis Management).
*   **SINDH:** Historical media archive/library, automated scheduled reporting.

---

## 2. Partially Overlapping Features
*These features are shared by at least two clients but differ significantly in implementation methods, scope, or target metrics.*

### A. Advertisement Tracking vs. Ad Detection
*   **DGPR (Classification):** Focuses on ad segment classification and identification (e.g., separating commercial content from news feed for compliance).
*   **FOCUS 360 (Campaign Analysis & ROI):** Tracks reach, impressions, engagement rates, and ROI for social media, print, and electronic ads.
*   **SINDH (Acoustic Matching):** Detects specific 90-110 second TV and radio ad segments using audio fingerprinting and temporal matching.

### B. Computer Vision (Face, Logo & Object Detection)
*   **DGPR & SINDH (Full Visual AI):** Both require visual analytics including facial recognition, logo detection, and object/scene detection.
*   **FOCUS 360 (OCR-focused):** Focuses heavily on OCR (Tesseract, EasyOCR) for text extraction from print media, but lacks explicit requirements for live facial recognition or real-time object tracking in video feeds.

### C. Speaker Recognition
*   **DGPR & SINDH (Deep Speaker AI):** Both require deep learning models for speaker recognition/identification. DGPR adds Speaker Diarization (determining who spoke when), while SINDH focuses specifically on identifying key speakers, politicians, and influencers.
*   **FOCUS 360:** Does not specify any speaker recognition requirements.

### D. Print Media Monitoring & OCR
*   **FOCUS 360 & SINDH (Dedicated News OCR):** FOCUS 360 requires extracting text from scanned/digital newspapers for keyword matching. SINDH specifies monitoring exactly 50 digital print newspapers with OCR extraction.
*   **DGPR (General Document MAM):** Supports multi-format files (PDF, JPEG, PNG, DOCX) and general visual OCR, but doesn't explicitly frame it as a dedicated "print newspaper monitoring" pipeline.

### E. Frontend Interfaces & Mobile Support
*   **SINDH (Web & Mobile Apps):** Requires deployment across a Web Portal, Android App, and iOS App.
*   **DGPR & FOCUS 360 (Web Dashboards):** Only require responsive web-based dashboards and consoles (using React/Next.js/Angular/Vue).

---

## 3. Non-Overlapping Features
*These features are completely unique to a single client and represent custom requirements.*

### A. DGPR Unique Features
*   **Compliance Monitoring:** PEMRA code violation detection.
*   **Advanced Content Analytics:** Toxic/Hate speech detection, Profanity detection, Fake news detection.
*   **Political Insights:** Public opinion tracking, narrative shift detection, and daily political summaries.
*   **Media Editing:** Multi-point editing tools built into the Media Asset Management (MAM) system.
*   **Scale & Security Infrastructure:** High-scale infrastructure (100TB-200TB+ NVMe nodes, 10G/100G switches, and Zero-Trust architecture with VLAN segmentation).

### B. FOCUS 360 Unique Features
*   **Influencer Campaign Analysis:** Evaluates influencer collaborations based on ROI, engagement metrics, and audience demographics.
*   **Clipping Module:** Manual and automated video clip creation tools with a searchable clip library.
*   **Crisis Management Reporting:** Specialized crisis reporting dashboard and alerts.

### C. SINDH Unique Features
*   **Ticker-to-Text:** Real-time text extraction from scrolling video tickers for 25 selected channels.
*   **Regional Language Support:** Speech-to-text transcription specifically for Sindhi (3 channels) and Pashto (3 channels) in addition to Urdu/English.
*   **Fixed User Limit:** Explicit scaling limit of 120 access logins.
*   **In-App Sharing:** Native capabilities for sharing media and reports directly within the application interfaces.
