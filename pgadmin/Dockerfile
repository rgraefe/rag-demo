# the following commands are used for debug purposes only
FROM dpage/pgadmin4:latest

USER root
# ENV http_proxy=http://proxy-mu.intel.com:911
# ENV https_proxy=https://proxy-mu.intel.com:912
# ENV no_proxy=localhost,127.0.0.1,.intel.com
# # Update and upgrade packages before installing new ones
RUN apk update && apk upgrade && apk add --no-cache bash curl jq vim

