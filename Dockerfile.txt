# Gunakan image Python 3.10 sebagai base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Salin file requirements.txt dan install dependencies
COPY requirement.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh aplikasi ke dalam container
COPY . /app/

# Expose port 8080 (port default Cloud Run)
EXPOSE 8080

# Jalankan aplikasi
CMD ["python", "app.py"]
