# Build stage
FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package.json ./
COPY package-lock.json* ./

# Clean npm cache and install dependencies
# Use longer timeout and retries to tolerate flaky networks during docker build
RUN npm cache clean --force \
    && npm config set fetch-timeout 300000 \
    && npm config set fetch-retries 5 \
    && npm config set fetch-retry-mintimeout 20000 \
    && npm config set fetch-retry-maxtimeout 120000
RUN npm install --production=false

# Vite embeds VITE_* at build time
ARG VITE_API_URL=
ARG VITE_APP_NAME=Bastion
ARG VITE_VERSION=0.70.0
ARG VITE_MAP_TILE_URL=
ARG VITE_PMTILES_URL=/pmtiles/world.pmtiles
ARG VITE_MAP_DARK_STYLE=
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_APP_NAME=$VITE_APP_NAME
ENV VITE_VERSION=$VITE_VERSION
ENV VITE_MAP_TILE_URL=$VITE_MAP_TILE_URL
ENV VITE_PMTILES_URL=$VITE_PMTILES_URL
ENV VITE_MAP_DARK_STYLE=$VITE_MAP_DARK_STYLE

# Copy source code
COPY index.html ./
COPY vite.config.mjs ./
COPY public/ ./public/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env* ./

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
