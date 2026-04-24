# 1. Use the exact Python version that works perfectly with MediaPipe/TensorFlow
FROM python:3.11

# 2. Set up the working directory inside the server
WORKDIR /code

# 3. Install the missing Linux graphics library for OpenCV
RUN apt-get update && apt-get install -y libgl1

# 4. Copy your requirements file and install your Python packages
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy the rest of your app code into the server
COPY . .

# 6. Tell Docker to run Streamlit on the specific port Hugging Face requires (7860)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]