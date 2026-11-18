-- AirInk: database setup (created automatically on first run)
CREATE DATABASE IF NOT EXISTS airink_db;
USE airink_db;

-- One row per saved drawing. The PNG itself stays in the drawings folder.
CREATE TABLE IF NOT EXISTS drawings (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    title      VARCHAR(100) NOT NULL,
    file_name  VARCHAR(100) NOT NULL UNIQUE,
    theme      VARCHAR(30),
    board      VARCHAR(30),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- One row per drawing session (from opening the app to closing it).
CREATE TABLE IF NOT EXISTS sessions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    started_at  DATETIME NOT NULL,
    ended_at    DATETIME NOT NULL,
    strokes     INT DEFAULT 0,
    shapes      INT DEFAULT 0,
    flowers     INT DEFAULT 0,
    erases      INT DEFAULT 0,
    undos       INT DEFAULT 0,
    saves       INT DEFAULT 0,
    used_camera BOOLEAN DEFAULT FALSE
);
