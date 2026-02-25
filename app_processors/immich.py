# External Libraries (Easiest - Zero Scripts)
# Mount a folder to Immich and enable automatic file watching - new files are automatically imported without rescanning
# Setup:
#
# In your docker-compose.yml:
#
# yamlimmich-server:
#   volumes:
#     - ${UPLOAD_LOCATION}:/data
#     - /path/to/your/photos:/mnt/watch:ro  # Add this
#
# In Immich web UI:
#
# Go to Administration → Libraries
# Create External Library
# Point to /mnt/watch
# Enable "Watch for filesystem changes" Immich
#
#
# Drop files into /path/to/your/photos → Immich auto-imports them
#
# Note: If photos are on a network drive, automatic watching likely won't work - you'd need periodic rescans instead
