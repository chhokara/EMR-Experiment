FROM public.ecr.aws/emr-on-eks/spark/emr-6.14.0:latest

RUN python3 -m ensurepip --upgrade && \
    pip3 install --upgrade pip && \
    pip3 install numpy


    