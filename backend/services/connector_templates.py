"""
Pre-built connector templates for Agent Factory data sources.

Templates are used by the UI to create new data_source_connectors (user copies
the definition and can attach credentials). Execution runs in connections-service via
connector_executor.execute_connector (REST, web_fetch, sftp, s3, webdav).
"""

# Template shape: name, description, connector_type, definition, requires_auth, auth_fields, icon, category

CONNECTOR_TEMPLATES = [
    {
        "name": "JSONPlaceholder (Public REST)",
        "description": "Sample public REST API for testing. No auth required.",
        "connector_type": "rest",
        "requires_auth": False,
        "auth_fields": [],
        "icon": "api",
        "category": "Public API",
        "definition": {
            "base_url": "https://jsonplaceholder.typicode.com",
            "auth": {},
            "endpoints": {
                "list_posts": {
                    "path": "/posts",
                    "method": "GET",
                    "params": [],
                    "response_list_path": ".",
                    "description": "List all posts",
                },
                "get_post": {
                    "path": "/posts/{id}",
                    "method": "GET",
                    "params": [{"name": "id", "in": "path", "description": "Post ID"}],
                    "response_list_path": ".",
                    "description": "Get a single post by ID",
                },
                "list_users": {
                    "path": "/users",
                    "method": "GET",
                    "params": [],
                    "response_list_path": ".",
                    "description": "List all users",
                },
            },
        },
    },
    {
        "name": "Web Fetch (URL to JSON)",
        "description": "Fetch a URL and return response as JSON. Use for public JSON APIs or file URLs.",
        "connector_type": "web_fetch",
        "requires_auth": False,
        "auth_fields": [],
        "icon": "link",
        "category": "Web",
        "definition": {
            "base_url": "",
            "auth": {},
            "endpoints": {
                "fetch": {
                    "path": "{url}",
                    "method": "GET",
                    "params": [{"name": "url", "in": "path", "description": "Full URL to fetch (e.g. https://api.example.com/data.json)", "required": True}],
                    "response_list_path": ".",
                    "description": "GET URL and return JSON body",
                },
            },
        },
    },
    {
        "name": "OpenFEC (Federal Election Commission)",
        "description": "US Federal Election Commission API. Requires free API key from https://api.open.fec.gov/.",
        "connector_type": "rest",
        "requires_auth": True,
        "auth_fields": [{"name": "api_key", "label": "API Key", "type": "password"}],
        "icon": "election",
        "category": "Public API",
        "definition": {
            "base_url": "https://api.open.fec.gov/v1",
            "auth": {"type": "api_key", "credentials_key": "api_key", "header_name": "X-Api-Key"},
            "endpoints": {
                "committees": {
                    "path": "/committees",
                    "method": "GET",
                    "params": [
                        {"name": "committee_id", "in": "query", "description": "Optional committee ID filter"},
                        {"name": "per_page", "in": "query", "default": 20},
                    ],
                    "response_list_path": "results",
                    "pagination": {"type": "page", "page_param": "page"},
                    "description": "List committees",
                },
                "candidates": {
                    "path": "/candidates",
                    "method": "GET",
                    "params": [
                        {"name": "candidate_id", "in": "query", "description": "Optional candidate ID filter"},
                        {"name": "per_page", "in": "query", "default": 20},
                    ],
                    "response_list_path": "results",
                    "pagination": {"type": "page", "page_param": "page"},
                    "description": "List candidates",
                },
            },
        },
    },
    {
        "name": "Hacker News API",
        "description": "Hacker News public API. No auth required.",
        "connector_type": "rest",
        "requires_auth": False,
        "auth_fields": [],
        "icon": "article",
        "category": "Public API",
        "definition": {
            "base_url": "https://hacker-news.firebaseio.com/v0",
            "auth": {},
            "endpoints": {
                "top_stories": {
                    "path": "/topstories.json",
                    "method": "GET",
                    "params": [],
                    "response_list_path": ".",
                    "description": "List top story IDs",
                },
                "get_item": {
                    "path": "/item/{id}.json",
                    "method": "GET",
                    "params": [{"name": "id", "in": "path", "description": "Item ID", "required": True}],
                    "response_list_path": ".",
                    "description": "Get item by ID",
                },
            },
        },
    },
    {
        "name": "GitHub REST API",
        "description": "GitHub API. Requires personal access token.",
        "connector_type": "rest",
        "requires_auth": True,
        "auth_fields": [{"name": "token", "label": "Personal Access Token", "type": "password"}],
        "icon": "code",
        "category": "Developer",
        "definition": {
            "base_url": "https://api.github.com",
            "auth": {"type": "bearer", "credentials_key": "token"},
            "endpoints": {
                "user_repos": {
                    "path": "/user/repos",
                    "method": "GET",
                    "params": [{"name": "per_page", "in": "query", "default": 30}],
                    "response_list_path": ".",
                    "description": "List user repos",
                },
                "repo_issues": {
                    "path": "/repos/{owner}/{repo}/issues",
                    "method": "GET",
                    "params": [
                        {"name": "owner", "in": "path", "required": True},
                        {"name": "repo", "in": "path", "required": True},
                        {"name": "state", "in": "query", "default": "open"},
                    ],
                    "response_list_path": ".",
                    "description": "List repo issues",
                },
            },
        },
    },
    {
        "name": "Alpha Vantage",
        "description": "Stock and forex data from Alpha Vantage. Requires free API key from https://www.alphavantage.co/support/#api-key. Key is sent as a query parameter.",
        "connector_type": "rest",
        "requires_auth": True,
        "auth_fields": [{"name": "api_key", "label": "API Key", "type": "password"}],
        "icon": "trending_up",
        "category": "Finance",
        "definition": {
            "base_url": "https://www.alphavantage.co",
            "auth": {"type": "api_key", "credentials_key": "api_key", "location": "query", "param_name": "apikey"},
            "endpoints": {
                "global_quote": {
                    "path": "/query",
                    "method": "GET",
                    "params": [
                        {"name": "function", "value": "GLOBAL_QUOTE", "in": "query"},
                        {"name": "symbol", "in": "query", "required": True, "description": "Stock symbol (e.g. IBM, AAPL)"},
                    ],
                    "response_list_path": "Global Quote",
                    "description": "Get latest quote for a symbol",
                },
                "time_series_intraday": {
                    "path": "/query",
                    "method": "GET",
                    "params": [
                        {"name": "function", "value": "TIME_SERIES_INTRADAY", "in": "query"},
                        {"name": "symbol", "in": "query", "required": True, "description": "Stock symbol"},
                        {"name": "interval", "in": "query", "default": "15min", "description": "1min, 5min, 15min, 30min, 60min"},
                        {"name": "outputsize", "value": "compact", "in": "query"},
                    ],
                    "response_list_path": "Time Series (15min)",
                    "description": "Intraday time series (default 15min)",
                },
            },
        },
    },
    {
        "name": "SFTP Server",
        "description": "SSH SFTP file access. Set host and base_path; use password or private_key PEM in credentials.",
        "connector_type": "sftp",
        "requires_auth": True,
        "auth_fields": [
            {"name": "username", "label": "Username", "type": "text"},
            {"name": "password", "label": "Password", "type": "password"},
            {"name": "private_key", "label": "Private key (PEM)", "type": "textarea"},
            {"name": "passphrase", "label": "Key passphrase", "type": "password"},
        ],
        "icon": "folder",
        "category": "File transfer",
        "definition": {
            "connector_type": "sftp",
            "host": "",
            "port": 22,
            "base_path": "/",
            "endpoints": {
                "list_files": {
                    "operation": "list",
                    "path": "{remote_path}",
                    "defaults": {"remote_path": "."},
                    "description": "List directory; params: remote_path (default .)",
                },
                "read_file": {
                    "operation": "read",
                    "path": "{remote_path}",
                    "description": "Read file as base64; params: remote_path",
                },
                "write_file": {
                    "operation": "write",
                    "path": "{remote_path}",
                    "description": "Write file; params: remote_path, content_base64 or content_text",
                },
                "delete_file": {
                    "operation": "delete",
                    "path": "{remote_path}",
                    "description": "Delete file; params: remote_path",
                },
                "mkdir": {
                    "operation": "mkdir",
                    "path": "{remote_path}",
                    "description": "Create directory tree; params: remote_path",
                },
            },
        },
    },
    {
        "name": "S3 / S3-Compatible Storage",
        "description": "AWS S3, MinIO, Cloudflare R2, etc. Set bucket, region, and optional endpoint_url for non-AWS.",
        "connector_type": "s3",
        "requires_auth": True,
        "auth_fields": [
            {"name": "access_key_id", "label": "Access key ID", "type": "text"},
            {"name": "secret_access_key", "label": "Secret access key", "type": "password"},
            {"name": "session_token", "label": "Session token (optional)", "type": "password"},
        ],
        "icon": "cloud",
        "category": "File transfer",
        "definition": {
            "connector_type": "s3",
            "bucket": "",
            "region": "us-east-1",
            "endpoint_url": "",
            "prefix": "",
            "endpoints": {
                "list_objects": {
                    "operation": "list",
                    "prefix": "{prefix}",
                    "defaults": {"prefix": ""},
                    "description": "List objects; params: prefix (optional, defaults to definition prefix)",
                },
                "get_object": {
                    "operation": "read",
                    "key": "{key}",
                    "description": "Get object as base64; params: key",
                },
                "put_object": {
                    "operation": "write",
                    "key": "{key}",
                    "description": "Put object; params: key, content_base64 or content_text, optional content_type",
                },
                "delete_object": {
                    "operation": "delete",
                    "key": "{key}",
                    "description": "Delete object; params: key",
                },
            },
        },
    },
    {
        "name": "WebDAV Server",
        "description": "WebDAV over HTTPS. Set base_url and optional base_path; HTTP Basic auth.",
        "connector_type": "webdav",
        "requires_auth": True,
        "auth_fields": [
            {"name": "username", "label": "Username", "type": "text"},
            {"name": "password", "label": "Password", "type": "password"},
        ],
        "icon": "cloud_sync",
        "category": "File transfer",
        "definition": {
            "connector_type": "webdav",
            "base_url": "https://",
            "base_path": "",
            "endpoints": {
                "list_files": {
                    "operation": "list",
                    "path": "{remote_path}",
                    "defaults": {"remote_path": "."},
                    "description": "PROPFIND Depth 1; params: remote_path (directory, default .)",
                },
                "read_file": {
                    "operation": "read",
                    "path": "{remote_path}",
                    "description": "GET file; params: remote_path",
                },
                "write_file": {
                    "operation": "write",
                    "path": "{remote_path}",
                    "description": "PUT file; params: remote_path, content_base64 or content_text",
                },
                "delete_file": {
                    "operation": "delete",
                    "path": "{remote_path}",
                    "description": "DELETE; params: remote_path",
                },
                "mkdir": {
                    "operation": "mkdir",
                    "path": "{remote_path}",
                    "description": "MKCOL collection; params: remote_path",
                },
            },
        },
    },
    {
        "name": "Ghost Admin API",
        "description": "Ghost CMS Admin API. Admin API key from Ghost Admin > Integrations > Custom integration (format id:secret). Set base_url to your site origin (no trailing path).",
        "connector_type": "rest",
        "requires_auth": True,
        "auth_fields": [
            {"name": "admin_api_key", "label": "Admin API Key (id:secret)", "type": "password"},
        ],
        "icon": "article",
        "category": "Publishing",
        "definition": {
            "base_url": "https://your-ghost-site.com",
            "headers": {
                "Accept-Version": "v5.0",
            },
            "auth": {
                "type": "jwt",
                "algorithm": "HS256",
                "compound_key_field": "admin_api_key",
                "compound_key_separator": ":",
                "secret_encoding": "hex",
                "header_prefix": "Ghost",
                "claims": {"aud": "/admin/"},
                "exp_seconds": 300,
            },
            "endpoints": {
                "list_posts": {
                    "path": "/ghost/api/admin/posts/",
                    "method": "GET",
                    "params": [
                        {"name": "limit", "in": "query", "default": 15},
                        {"name": "page", "in": "query", "default": 1},
                    ],
                    "response_list_path": "posts",
                    "pagination": {"type": "page", "page_param": "page"},
                    "description": "Browse posts; query: limit, page",
                },
                "get_post": {
                    "path": "/ghost/api/admin/posts/{id}/",
                    "method": "GET",
                    "params": [{"name": "id", "in": "path", "required": True, "description": "Post ID"}],
                    "response_list_path": "posts",
                    "description": "Read one post by id",
                },
                "create_post": {
                    "path": "/ghost/api/admin/posts/",
                    "method": "POST",
                    "params": [
                        {"name": "title", "in": "query", "required": True, "description": "Post title"},
                        {"name": "html", "in": "query", "default": "", "description": "HTML body"},
                        {"name": "status", "in": "query", "default": "draft", "description": "draft|published|scheduled"},
                    ],
                    "body_template": {"posts": [{"title": "{title}", "html": "{html}", "status": "{status}"}]},
                    "response_list_path": "posts",
                    "description": "Create post; params: title, html, status",
                },
                "upload_image": {
                    "path": "/ghost/api/admin/images/upload/",
                    "method": "POST",
                    "body_mode": "multipart_file",
                    "multipart_field_name": "file",
                    "params": [
                        {"name": "file_base64", "in": "query", "required": True, "description": "Image bytes as base64"},
                        {"name": "filename", "in": "query", "required": True, "description": "Original filename e.g. photo.png"},
                        {"name": "content_type", "in": "query", "default": "image/png", "description": "MIME type"},
                    ],
                    "response_list_path": "images",
                    "description": "Upload image; params: file_base64, filename, optional content_type",
                },
            },
        },
    },
]
