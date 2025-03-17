#!/bin/bash

# 設置變量
DOMAIN="10.0.239.87"
CERT_DIR="./ssl"
DAYS_VALID=365

# 創建 SSL 目錄（如果不存在）
mkdir -p $CERT_DIR

# 生成私鑰
openssl genrsa -out $CERT_DIR/private.key 2048

# 生成證書簽名請求（CSR）
openssl req -new -key $CERT_DIR/private.key -out $CERT_DIR/cert.csr -subj "/CN=$DOMAIN/O=Local Development/C=TW"

# 生成自簽名證書
openssl x509 -req -days $DAYS_VALID -in $CERT_DIR/cert.csr -signkey $CERT_DIR/private.key -out $CERT_DIR/certificate.crt

# 刪除 CSR（不再需要）
rm $CERT_DIR/cert.csr

echo "自簽名證書已生成完成！"
echo "證書位置: $CERT_DIR/certificate.crt"
echo "私鑰位置: $CERT_DIR/private.key" 