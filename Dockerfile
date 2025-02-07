# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# (Optional) Install any additional system dependencies if required
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the local source code into the container (including your module)
COPY . .

# Install your local module (this assumes you have a setup.py or pyproject.toml in the root)
RUN pip install .

# (Optional) Expose a port if your application listens on one (e.g., for a web service)
# EXPOSE 8000

# (Optional) Set a default command; adjust this as needed for your application.
# For example, to run a main.py script:
# CMD ["python", "main.py"]

# to train the model make go to the docker shell
# docker exec -it {docker_name} bash