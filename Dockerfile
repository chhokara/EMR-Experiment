FROM public.ecr.aws/emr-on-eks/spark/emr-6.14.0:latest
USER root
RUN pip3 install numpy pandas pyspark
USER hadoop:hadoop 