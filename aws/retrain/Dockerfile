# Build context is the repo root (see template.yaml DockerContext: .)
FROM public.ecr.aws/lambda/python:3.12

COPY aws/retrain/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# train.py provides clean(); app.py retrains from S3 data
COPY train.py aws/retrain/app.py ${LAMBDA_TASK_ROOT}/

CMD ["app.lambda_handler"]
