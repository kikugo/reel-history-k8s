# syntax=docker/dockerfile:1
FROM node:22-slim AS build
WORKDIR /fe
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY index.html vite.config.js ./
COPY src ./src
RUN npm run build

# nginx-unprivileged runs as uid 101 and listens on 8080
FROM nginxinc/nginx-unprivileged:1.29-alpine
ENV RECOMMENDER_URL=http://recommender:8000 EXPLAINER_URL=http://explainer:8000
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /fe/dist /usr/share/nginx/html
EXPOSE 8080
