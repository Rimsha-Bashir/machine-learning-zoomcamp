## 9.1 AWS Lamda

- Create a function in lambda - select specfications.An IDE like window opens up for you to code your function
- Click on Test dropdown to define and declare parameters for 'event'
- Click on Test/Deploy to see the results.

![alt text](./images/9.1.1.png)

We don't need to create EC2 instances etc, since lambda is serverless. And we're charged only when the lambda function is doing something (event sent/triggered)

## 9.2 Deploy ML models with AWS Lambda 

FOLLOW - https://github.com/alexeygrigorev/workshops/tree/main/mlzoomcamp-serverless

1. Created an AWS account. 
2. Set up Amazon CLI (download msi file, save path)
```bash
rimsh@LAPTOP-J29FGN6B MINGW64 ~
$ aws --version
aws-cli/2.32.12 Python/3.13.9 Windows/10 exe/AMD64
```
3. Create a user, grant it necessary access and create an access key for it. (In order to safely perform the next steps)

**AWS expects programmatic access to always use an IAM user, never root.**

4. Then run `aws configure`

```bash
rimsh@LAPTOP-J29FGN6B MINGW64 ~
$ aws configure
AWS Access Key ID [None]: enter here
AWS Secret Access Key [None]: enter here
Default region name [None]: us-east-1
Default output format [None]: json

```
>:bulb: All of this set up is required to connect to AWS from local. Why?
>
>    1. Your script is not local only — it calls AWS Lambda
>
>    2. AWS needs to authenticate who is calling (Access Keys)
>
>    3. AWS needs to authorize you (IAM permissions)
>
>    4. AWS needs to know where the Lambda is (region)
>
>    5. The AWS CLI is the easiest way to store this info so boto3 can find it
>
>    6. In short: this setup saves your script from failing every time it tries to access AWS.

2. Using uv instead of pipenv.
3. Install `pip install boto3`
4. To invoke the AWS function, you can either run the python invoke file (boto3), or run via the CLI using the command mentioned below.   

A. Method 1 - Using boto (invocation via invoke.py)

```bash
python invoke.py
```

Output:

```bash
{
  "churn_probability": 0.56,
  "churn": true
}
```


B. Method 2 - Using CLI (direct invokation)

```bash 
    rimsh@LAPTOP-J29FGN6B MINGW64 ~/Desktop/rimsha/github/machine-learning-zoomcamp/workshop/sklearn (main)
    $ aws lambda invoke \
    --region eu-north-1 \
    --function-name churn-prediction \
    --cli-binary-format raw-in-base64-out \
    --payload file://customer.json \
    output.json
```
```bash
    {
        "StatusCode": 200,
        "ExecutedVersion": "$LATEST"
    }
```

Status Code - 200, successfully executed

If you now do `ls`

you'll see 

```bash
rimsh@LAPTOP-J29FGN6B MINGW64 ~/Desktop/rimsha/github/machine-learning-zoomcamp/workshop/sklearn (main)
$ ls
customer.json  invoke.py  output.json  train/
```

Then, do 

```bash
cat output.json
```

Output:

```bash
{"churn_probability": 0.56, "churn": true}
```

Now, that a test (hardcoded version) is available, we move on to creating a `lambda_function.py` in local and then copying it to churn-prediction function (lambda_function.py) in AWS. 

5. Create lambda_function.py 

```python
import pickle


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

def lambda_handler(event, context):    
    print("Parameters:", event)
    customer = event['customer']
    prob = predict_single(customer)
    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }

```

But we can not simply put it in AWS, we need to install dependencies - so we need a docker container that will have all the dependencies installed along with the lambda function and then deploy it. 

Check out the [Dockerfile](../workshop/sklearn/Dockerfile)

```bash 
$ docker build --no-cache -t churn-prediction-lambda .

$ docker run -it --rm --entrypoint bash churn-prediction-lambda
```

```bash
bash-5.2# ls
lambda_function.py  model.bin  pyproject.toml  uv.lock
bash-5.2# cat /var/task/lambda_function.py
import pickle


with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }bash-5.2# exit
exit
```
```bash
$ docker run -it --rm -p 8080:8080 churn-prediction-lambda 
```
Then, in another git bash window, 

run:

```bash
$ python test.py
```


Now, we need to publish docker image to ECR - elastic container registry! (where we store images)

Go to ECR and create a repository, call it ../churn-prediction-lambda 

No other settings to be changed

Create Repository, then copy URI 

We create a [publish.sh](../workshop/sklearn/publish.sh) file to do this. Check out the file. 

Testing the login:

```bash
rimsh@LAPTOP-J29FGN6B MINGW64 ~/Desktop/rimsha/github/machine-learning-zoomcamp/workshop/sklearn (main)
$ ECR_URL=565919381802.dkr.ecr.eu-north-1.amazonaws.com

rimsh@LAPTOP-J29FGN6B MINGW64 ~/Desktop/rimsha/github/machine-learning-zoomcamp/workshop/sklearn (main)
$ REPO_URL=${ECR_URL}/churn-prediction-lambda

rimsh@LAPTOP-J29FGN6B MINGW64 ~/Desktop/rimsha/github/machine-learning-zoomcamp/workshop/sklearn (main)
$ LOCAL_IMAGE=churn-prediction-lambda

rimsh@LAPTOP-J29FGN6B MINGW64 ~/Desktop/rimsha/github/machine-learning-zoomcamp/workshop/sklearn (main)
$ aws ecr get-login-password \
>   --region "eu-north-1" \
> | docker login \
>   --username AWS \
>   --password-stdin ${ECR_URL}
Login Succeeded
```

After login succeeds, we proceed to tag and push the docker image to the registry. 

`docker push command`: to share your images to the Docker Hub registry or to a self-hosted one. 
`docker tag command`: a Docker image tag is simply a label that helps you identify a specific version of a Docker image.

Then, create a new lambda function and use the above image. 

Test with a `customer` json input. 


### :exclamation::exclamation: Skipping ONNX (Open Neural Network Exchange) for now. Check out the [workshop video (second half)](https://www.youtube.com/watch?v=sHQaeVm5hT8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=84) to learn how to use it. The notes are [here](https://github.com/alexeygrigorev/workshops/tree/main/mlzoomcamp-serverless)


## NOTES:

1. What is boto3?

- **Boto3** = AWS SDK for Python (handles authentication, requests, services)

- GCP equivalent = **google-cloud** Python libraries (service-specific clients, use service account JSON for auth)

Both SDKs allow programmatic access to cloud services from Python.

For more information check out [ML Zoomcamp Workshops repository](https://github.com/alexeygrigorev/workshops)

2. uv init initializes the dependencies (it's a fast py packet manager)