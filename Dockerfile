FROM ubuntu:18.04


# install all the dependency
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	apt-utils \
	build-essential \
	gfortran \
	cmake \
	git \
	wget \
	pkg-config \
	libgsl-dev \
	libblas-dev \
	liblapack-dev \
	libsqlite3-dev \
	libsuitesparse-dev \
	python3 \
	python3-pip \
	python3-dev \
	python3-numpy \
	python3-pandas \
	python3-scipy \
	python3-xlrd \
	python3-setuptools \
	python3-matplotlib \
	vim \
	netbase


# install jupyter notebook
RUN pip3 install jupyter


# clone the refered version of dismod_at repository
WORKDIR /home
RUN git clone https://github.com/bradbell/dismod_at.git
WORKDIR /home/dismod_at
RUN git pull
RUN git checkout bd34020c4f01083ad257925ef83abd9dcc107488

RUN mkdir /home/prefix
RUN sed -i bin/run_cmake.sh -e 's|$HOME/|/home/|g'


# install debug version of dismod_at
RUN sed -i bin/run_cmake.sh -e "s|^build_type=.*|build_type='debug'|"
RUN bin/example_install.sh


# install release version of dismod_at
RUN sed -i bin/run_cmake.sh -e "s|^build_type=.*|build_type='release'|"
RUN bin/example_install.sh


# remove the repo to save space
WORKDIR /home
RUN rm -rf dismod_at.git


# add rcfile for the use of Singularity image
#RUN touch /home/setup_env.sh
#RUN echo "export PATH=\"/home/prefix/dismod_at.\${BUILD_TYPE}/bin:\${PATH}\"" \
#	>> /home/setup_env.sh
#RUN echo "export LD_LIBRARY_PATH=\"/home/prefix/dismod_at.\${BUILD_TYPE}/lib64:\${LD_LIBRARY_PATH}\"" \
#	>> /home/setup_env.sh
#RUN echo "export PKG_CONFIG_PATH=\"/home/prefix/dismod_at.\${BUILD_TYPE}/lib64/pkgconfig\"" \
#	>> /home/setup_env.sh
#RUN echo "export PYTHONPATH=\"/home/prefix/dismod_at.\${BUILD_TYPE}/lib/python3.6/site-packages\"" \
#	>> /home/setup_env.sh

ENV BUILD_TYPE="release"
ENV PATH="/home/prefix/dismod_at.${BUILD_TYPE}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/home/prefix/dismod_at.${BUILD_TYPE}/lib64:${LD_LIBRARY_PATH}"
ENV PKG_CONFIG_PATH="/home/prefix/dismod_at.${BUILD_TYPE}/lib64/pkgconfig"
ENV PYTHONPATH="/home/prefix/dismod_at.${BUILD_TYPE}/lib/python3.6/site-packages"


# start in the /home/work directory
WORKDIR /home/work
