FROM public.ecr.aws/lambda/python:3.13
WORKDIR /var/task

# Install uv CLI
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Install bash so you can inspect the container
COPY pyproject.toml uv.lock ./ 

RUN uv pip install --system -r <(uv export --format requirements-txt)

# Copy Lambda code and model into /var/task
COPY ./lambda_function.py ./model.bin /var/task/

# USING ABSOLUTE PATHS!! instead of using lambda_function.py use ./lambda_function.py
# ./ explicitly tells Docker: “This file is in the root of the build context.”


# Set Lambda handler
CMD ["lambda_function.lambda_handler"]
