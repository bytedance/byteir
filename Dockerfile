FROM klakegg/hugo:ext-debian

WORKDIR /app
COPY ./ /app

CMD ["server"]
