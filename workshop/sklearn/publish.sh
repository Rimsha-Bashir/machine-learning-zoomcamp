ECR_URL=565919381802.dkr.ecr.eu-north-1.amazonaws.com
REPO_URL=${ECR_URL}/churn-prediction-lambda
LOCAL_IMAGE=churn-prediction-lambda

docker build -t ${LOCAL_IMAGE} .

aws ecr get-login-password \
  --region "eu-north-1" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}