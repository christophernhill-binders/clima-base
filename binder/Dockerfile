# Build using
#  docker build -t clima -f CLIMA.dockerfile .

# Typical run command
#  docker run -d -v `pwd`:/home/jlab/host -P -i -t --rm clima /bin/bash



# FROM julia@sha256:c13bc0b99ab003de993078fdf70481bc0fd500ebf1d38968d89d32db6905a446
FROM jupyter/datascience-notebook:latest

USER root
RUN apt-get update && apt-get install -y gfortran mpich libmpich-dev less vim wget bzip2 procps git net-tools screen cmake ffmpeg xorg-dev sudo

RUN ( adduser --disabled-password --shell /bin/bash --gecos "User" juser; usermod -aG sudo juser; passwd -d juser )
RUN ( usermod -aG sudo juser; passwd -d jovyan )

# USER juser

# ENV JROOT=/home/juser
ENV CGIT=https://github.com/climate-machine/CLIMA.git
ENV CC=/usr/bin/mpicc
ENV CXX=/usr/bin/mpicxx
ENV FC=/usr/bin/mpif90
