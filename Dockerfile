# Specify the parent image from which we build
FROM python/python:3.9.18

# Set the working directory inside the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

# Define environment variable (if needed)
ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY . /app/.
RUN pip install --no-cache-dir -r /app/requirements.txt


# Run multiple_face_detect_zed.py when the container launches
CMD ["python3", "main.py"]
