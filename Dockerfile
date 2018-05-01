#Download base image ubuntu 16.04
FROM ubuntu:latest

# Recursively add all files within home directory
ADD . /

# Install all required dependencies (sklearn dependent on numpy and scipy)
RUN apt-get install && apt-get update
RUN apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran python3-pip
RUN pip3 install --upgrade numpy scipy
RUN pip3 install --upgrade -r requirements.txt

# Configure source
EXPOSE 80

RUN ["chmod", "+x", "/usr/lib/python3"]
CMD [ "python3", "./run_bot.py", "--token=TELEGRAM_API_KEY"]
