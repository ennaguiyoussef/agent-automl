# Agentic AutoML — frontend image
# Stage 1: build the static bundle with Vite.
FROM node:20-alpine AS build

WORKDIR /app

# Install deps from the lockfile (cached unless package files change).
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# The app reads VITE_API_BASE_URL at build time. Left empty so the bundle uses
# a relative "/api" base — nginx (below) proxies that to the backend, keeping
# everything same-origin (no CORS in production).
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# Stage 2: serve with nginx.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
