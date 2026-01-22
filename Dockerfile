FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 5001

# Command to run the model training script
CMD ["python", "src/app.py"]