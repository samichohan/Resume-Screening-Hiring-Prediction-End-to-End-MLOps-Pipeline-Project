# Base image — Python 3.10 use karo
FROM python:3.10-slim

# Working directory set karo
WORKDIR /app

# Requirements copy karo aur install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Poora project copy karo
COPY . .

# Port expose karo
EXPOSE 8501

# Streamlit chalao
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]