# =========================================================
# Stage 1: Build Stage - Compile React / Vite SPA bundle
# =========================================================
FROM node:20-alpine AS build-stage

WORKDIR /app

# Copy package config files
COPY frontend/package.json ./

# Install Node modules
RUN npm install

# Copy frontend source code
COPY frontend/ ./

# Declare build argument for API URL (configurable at build time)
# Default value points to local FastAPI backend service
ARG VITE_API_URL=http://127.0.0.1:8000
ENV VITE_API_URL=${VITE_API_URL}

# Build static production bundle into /app/dist
RUN npm run build

# =========================================================
# Stage 2: Production Stage - Lightweight Nginx web server
# =========================================================
FROM nginx:alpine AS production-stage

# Copy static build output from Stage 1 into Nginx public web root
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Expose HTTP port 80
EXPOSE 80

# Run Nginx in foreground mode
CMD ["nginx", "-g", "daemon off;"]
