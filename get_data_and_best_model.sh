wget https://storage.yandexcloud.net/number-recognition/data.zip
unzip data.zip
rm data.zip

mkdir checkpoints

wget https://storage.yandexcloud.net/number-recognition/best-checkpoint.ckpt
mv best-checkpoint.ckpt checkpoints