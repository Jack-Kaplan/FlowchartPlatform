# Use the official Nginx image
FROM nginx:alpine

# Copy the frontend files into Nginx's default directory
COPY frontend/ /usr/share/nginx/html

# Expose port 80 for HTTP traffic
EXPOSE 80
