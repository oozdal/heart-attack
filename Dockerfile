# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Copy all files from the current directory to the working directory in the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "streamlit_app.py"]
