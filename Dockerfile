FROM python:3.9-slim


WORKDIR /app

COPY ["requirements.txt","./"] 
RUN pip install -r requirements.txt

COPY ["predict.py","xception_v4_1_01_0.676.h5", "./"] 

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

