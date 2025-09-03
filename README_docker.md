
`docker buildx build -t qxy_app:apple_silicon .`

`docker buildx build --platform=linux/amd64 -t qxy_app:latest .`

`docker tag qxy_app:latest xyyandxyy/qxy_app:latest`

`docker push xyyandxyy/qxy_app:latest`