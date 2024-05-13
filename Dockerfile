# Use the official Python image as a base
FROM python:3.9

# Set the working directory in the container
WORKDIR /patent

# Copy the current directory contents into the container at /patent
COPY . /patent

# Install any needed packages specified in requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

