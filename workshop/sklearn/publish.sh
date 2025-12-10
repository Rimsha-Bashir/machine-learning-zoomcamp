ECR_URL=565919381802.dkr.ecr.eu-north-1.amazonaws.com
REPO_URL=${ECR_URL}/churn-prediction-lambda
LOCAL_IMAGE=churn-prediction-lambda
REMOTE_IMAGE_TAG="${REPO_URL}:v1"

# Login to ECR
aws ecr get-login-password \
  --region "eu-north-1" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

# Build a single-architecture image
docker buildx build \
  --platform linux/amd64 \
  --load \
  --provenance=false \
  -t ${LOCAL_IMAGE} . 


# Tag and push
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "Docker Image published successfully."
