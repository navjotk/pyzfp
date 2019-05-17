FROM ubuntu:xenial


# Add the ubuntu-toolchain-r test ppa
RUN echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu xenial main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-ppa-xenial.list

# Import the Launchpad PPA public key
RUN gpg --keyserver keyserver.ubuntu.com --recv 1E9377A2BA9EF27F
RUN gpg --export --armor BA9EF27F | apt-key add -

# Ensure UTF-8 environment is correct for documentation builds
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Use bash, not dash, or later conditionals fail
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash

# Upgrade to the most recent package set
RUN apt-get update
RUN apt-get -y dist-upgrade

# Needed for the conda and devito installs later
RUN apt-get -y install wget bzip2 git make

# Default gcc version to install
ARG gccVersion=8

# Use MPI?
ARG MPI_INSTALL=1
ENV MPI_INSTALL=$MPI_INSTALL

# Install gcc/g++
RUN apt-get -y install gcc-$gccVersion g++-$gccVersion

# Set up alternatives
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$gccVersion 10
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$gccVersion 10
RUN update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-$gccVersion 10
RUN update-alternatives --install /usr/bin/ar ar /usr/bin/gcc-ar-$gccVersion 10
RUN update-alternatives --install /usr/bin/nm nm /usr/bin/gcc-nm-$gccVersion 10
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-$gccVersion 10
RUN update-alternatives --install /usr/bin/ranlib ranlib /usr/bin/gcc-ranlib-$gccVersion 10
RUN if [ -f /usr/bin/gcov-dump-$gccVersion ] ; then \
      update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-$gccVersion 10 ; \
    fi
RUN if [ -f /usr/bin/gcov-tool-$gccVersion ] ; then \
      update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-$gccVersion 10 ; \
    fi

# Install MPI
RUN if [ "$MPI_INSTALL" == "1" ] ; then \
      apt-get install -y -q mpich libmpich-dev ; \
    fi

# Set up conda
RUN wget --no-verbose https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /usr/local/miniconda
ENV PATH=/usr/local/miniconda/bin:$PATH
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda

RUN adduser --disabled-password --gecos "" pyzfp

COPY . /home/pyzfp

RUN chown -R pyzfp /home/pyzfp

USER pyzfp
WORKDIR /home/pyzfp

RUN pip install -e .
