-- Run this once in MySQL (the app can also create it automatically)
CREATE DATABASE IF NOT EXISTS road_damage_db;
USE road_damage_db;

CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    original_image VARCHAR(255) NOT NULL,
    result_image VARCHAR(255) NOT NULL,
    location VARCHAR(255) DEFAULT '',
    damage_type VARCHAR(50) NOT NULL,      -- Pothole / Crack / Pothole + Crack / None
    pothole_count INT DEFAULT 0,
    crack_count INT DEFAULT 0,
    damage_percent FLOAT DEFAULT 0,        -- % of road area that is damaged
    severity VARCHAR(10) NOT NULL,         -- Low / Medium / High / None
    method VARCHAR(30) DEFAULT 'opencv',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
