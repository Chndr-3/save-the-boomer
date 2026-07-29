"""Classifier Lambda: POST /classify -> label, store to DynamoDB, SNS on fraud.

Reuses train.clean() + predict.classify() so cloud inference matches local.
"""
import datetime
import json
import os
import uuid

import boto3
import joblib

from predict import classify

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb")
sns = boto3.client("sns")

BUCKET = os.environ["MODEL_BUCKET"]
MODEL_KEY = os.environ.get("MODEL_KEY", "model.joblib")
TABLE = os.environ["TABLE_NAME"]
TOPIC = os.environ["SNS_TOPIC_ARN"]

_bundle = None  # cached across warm invocations; refreshes on cold start


def _model():
    global _bundle
    if _bundle is None:
        path = "/tmp/model.joblib"
        s3.download_file(BUCKET, MODEL_KEY, path)
        _bundle = joblib.load(path)
    return _bundle


def _resp(code, body):
    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _resp(400, {"error": "body must be JSON"})
    text = (body.get("text") or "").strip()
    if not text:
        return _resp(400, {"error": "missing 'text'"})

    label, prob = classify(text, _model())
    item = {
        "id": str(uuid.uuid4()),
        "text": text,
        "label": label,
        # store as strings — DynamoDB rejects native floats
        "fraud_prob": str(round(prob["fraud"], 4)),
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    ddb.Table(TABLE).put_item(Item=item)

    if label == "fraud":
        sns.publish(
            TopicArn=TOPIC,
            Subject="Save the Boomer: fraud SMS detected",
            Message=f"Fraud detected ({prob['fraud']:.0%} confidence):\n\n{text[:300]}",
        )

    return _resp(200, {
        "id": item["id"],
        "label": label,
        "probabilities": {k: round(v, 4) for k, v in prob.items()},
    })
