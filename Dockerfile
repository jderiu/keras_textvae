FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER Jan Deriu <deri@zahw.ch>

ARG THEANO_VERSION=rel-0.8.2
ARG TENSORFLOW_VERSION=0.8.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=1.0.3
ARG LASAGNE_VERSION=v0.1
ARG TORCH_VERSION=latest
ARG CAFFE_VERSION=master

#RUN echo -e "\n**********************\nNVIDIA Driver Version\n**********************\n" && \
#	cat /proc/driver/nvidia/version && \
#	echo -e "\n**********************\nCUDA Version\n**********************\n" && \
#	nvcc -V && \
#	echo -e "\n\nBuilding your Deep Learning Docker Image...\n"

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python3-dev \
		python3-pip \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

RUN apt-get upgrade

# Add SNI support to Python
RUN pip3 --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

#upgrade pip to newest
RUN python3 -m pip install --upgrade pip

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python3-numpy \
		python3-scipy \
		python3-nose \
		python3-h5py \
		python3-skimage \
		python3-matplotlib \
		python3-pandas \
		python3-sklearn \
		python3-sympy \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip3 --no-cache-dir install --upgrade ipython && \
	pip3 --no-cache-dir install \
		Cython \
		ipykernel \
		jupyter \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq \
		&& \
	python3 -m ipykernel.kernelspec


#install libgpuarray
RUN git clone https://github.com/Theano/libgpuarray.git
RUN cd libgpuarray && \
        mkdir Build && \
        cd Build && \
        cmake .. -DCMAKE_BUILD_TYPE=Release &&  \
        make && \
        make install && \
        cd .. && \
        python3 setup.py build && \
        python3 setup.py install && \
        ldconfig

#Install Theano
RUN pip3 install --no-cache-dir --upgrade Theano && \
	\
	echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
		\n[lib]\ncnmem=0.95 \
		\n[nvcc]\nfastmath=True \
		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
		\n[DebugMode]\ncheck_finite=1" \
	> /root/.theanorc

# Install TensorFlow
RUN pip3 install --no-cache-dir --upgrade tensorflow-gpu

# Install Keras
RUN pip3 install --no-cache-dir --upgrade keras
RUN pip3 install --no-cache-dir --upgrade nltk
RUN pip3 install --no-cache-dir --upgrade tqdm
RUN pip3 install --no-cache-dir --upgrade gensim

RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader stopwords


RUN mkdir /DLFramework

COPY vae_architectures /DLFramework/vae_architectures
COPY preprocessing_utils.py /DLFramework
COPY output_text.py /DLFramework
COPY data_loader.py /DLFramework
COPY main.py /DLFramework

WORKDIR /DLFramework

CMD python3 main.py -c config_vae.json
