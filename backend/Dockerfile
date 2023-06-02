FROM python:3.11
WORKDIR ./backend
ENV LISTEN_PORT=5000
EXPOSE 5000
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# Command to run on container start
COPY . .
CMD [ "python" , "app.py" ]