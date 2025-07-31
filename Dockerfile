# Use official lightweight Python image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy files
COPY . /app

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Hugging Face expects
ENV PORT=7860
EXPOSE $PORT

# Run the Flask app
CMD ["python", "app.py"]
