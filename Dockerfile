FROM ubuntu:19.10


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
	python3-distutils-extra \
	vim \
	netbase

RUN pip3 install jupyter
RUN pip3 install PyMySQL

# clone the referred version of dismod_at repository
WORKDIR /home
RUN rm -rf dismod_at
RUN git clone https://github.com/bradbell/dismod_at.git
WORKDIR /home/dismod_at
RUN git pull
RUN git checkout a506b941ce378ac670022f84c8c23ab0208d9d16

RUN mkdir /home/prefix
RUN sed -i bin/run_cmake.sh -e 's|$HOME/|/home/|g'


# install debug version of dismod_at
RUN sed -i bin/run_cmake.sh -e "s|^build_type=.*|build_type='debug'|"
RUN bin/example_install.sh


# install release version of dismod_at
RUN sed -i bin/run_cmake.sh -e "s|^build_type=.*|build_type='release'|"
RUN bin/example_install.sh

WORKDIR /home
RUN rm -rf dismod_at

# add jupyter config file
RUN mkdir /root/.jupyter
RUN touch /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8890" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py

ENV BUILD_TYPE="release"
ENV PATH="/home/prefix/dismod_at.${BUILD_TYPE}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/home/prefix/dismod_at.${BUILD_TYPE}/lib64:${LD_LIBRARY_PATH}"
ENV PKG_CONFIG_PATH="/home/prefix/dismod_at.${BUILD_TYPE}/lib64/pkgconfig"
ENV PYTHONPATH="/home/prefix/dismod_at.${BUILD_TYPE}/lib/python3.7/site-packages"


# start in the /home/work directory
WORKDIR /home/work
