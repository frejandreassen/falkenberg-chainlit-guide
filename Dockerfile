FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application
COPY . .

# Expose the port Chainlit runs on (default is 8000)
EXPOSE 8000

# Start the Chainlit app with host set to 0.0.0.0 and without opening a browser window
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "-h"]