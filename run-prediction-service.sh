env PYTHONUNBUFFERED=true gunicorn \
    --workers 4 \
    --timeout 60 \
    server.app:app -b 0.0.0.0:5000