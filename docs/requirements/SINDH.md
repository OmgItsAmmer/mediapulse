# AI Media Monitoring Project

## Project Features

* **Multi-Platform Access:** Service hosted for access via a Web Portal, Android App, and iOS App.
* **User & Access Management:** System supports 120 access logins. It features role-based access control (RBAC), account management, and activity logs. 
* **Extensive Media Coverage:** Real-time monitoring of 43 TV channels, 200 Twitter handles, 100 YouTube channels, 100 web pages, 50 digital print newspapers, and FM radio.
* **AI-Powered Processing & Analysis:**
    * **Transcription Services:** Speech-to-text conversion in Urdu, English (3 channels), Pashto (3 channels), and Sindhi (3 channels).
    * **Speaker Recognition:** Deep learning models identify key speakers, politicians, and influencers.
    * **Computer Vision:** Face detection for key individuals, alongside image and video analysis to detect logos and objects.
    * **Sentiment Analysis:** AI-driven classification of text, audio, and video content as positive, negative, or neutral.
    * **Named Entity Recognition (NER):** Extraction of people, places, and organizations from the media.
    * **Ticker-to-Text:** Text extraction from scrolling video tickers for 25 selected channels.
    * **Advertisement Detection:** Identification of 90-110 second TV and radio ad segments using audio fingerprinting and temporal matching.
* **Search and Visualization:** Dynamic search engine supporting text, speaker, and image queries. An analytics dashboard visualizes trends using summary graphs and heatmaps.
* **Alerts and Notifications:** Automated, real-time alert system triggered by event detection, specific keywords, and phrases.
* **Data Management & Reporting:** A media archive and library for historical data retrieval, combined with automated scheduled reporting.
* **Sharing & Integration:** Dynamic content sharing within the app and API capabilities for third-party and CRM system integrations.

---

## Tech Stack & Architecture

* **Data Ingestion & Extraction:** Kafka, FFmpeg, API integrations, and automated web crawlers/scrapers.
* **Database & Storage:** MongoDB for structured data storage.
* **AI & Machine Learning:** Custom trained models for face detection, deep learning models for speaker recognition, OCR technology for extracting embedded text/images, audio fingerprinting, and industry-leading LLMs tailored for sentiment analysis.
* **Infrastructure & Deployment:** On-premises hosting infrastructure combined with cloud-based architecture for dynamic scaling, load balancing, and automated failover mechanisms.
* **Security:** Encryption techniques for sensitive data, secure access logs, and compliance with GDPR and CCPA guidelines.
