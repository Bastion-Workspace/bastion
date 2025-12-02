# Build stage
FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package.json ./
COPY package-lock.json* ./

# Clean npm cache and install dependencies
RUN npm cache clean --force
RUN npm install --production=false

# Copy source code
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
