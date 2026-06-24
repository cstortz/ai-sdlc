-- Creates the Redmine database alongside the sdlc database.
-- Runs automatically on first Postgres container start (00-redmine-db.sql).
CREATE DATABASE redmine;
GRANT ALL PRIVILEGES ON DATABASE redmine TO sdlc;
