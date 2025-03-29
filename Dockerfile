FROM public.ecr.aws/emr-on-eks/spark/emr-6.14.0:latest

RUN pip3 install --upgrade boto3 pandas numpy
