-- Remove unused legacy GitHub org-mode sync schema (no application code referenced these tables).
DROP TABLE IF EXISTS github_issue_sync CASCADE;
DROP TABLE IF EXISTS github_project_mappings CASCADE;
DROP TABLE IF EXISTS github_connections CASCADE;
