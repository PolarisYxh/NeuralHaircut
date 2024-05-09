FROM colmap/colmap:latest
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ focal main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security multiverse\n"\ > /etc/apt/sources.list
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip git
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app
RUN mkdir ~/.pip&&echo '[global]\ntimeout = 6000\nindex-url = http://pypi.douban.com/simple\ntrusted-host = pypi.douban.com EOF' > ~/.pip/pip.conf