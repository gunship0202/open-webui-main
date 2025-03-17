# 設置變量
$domain = "10.0.239.87"
$sslPath = ".\ssl"
$certFile = "$sslPath\certificate.crt"
$keyFile = "$sslPath\private.key"
$pfxFile = "$sslPath\certificate.pfx"

# 創建 SSL 目錄（如果不存在）
New-Item -ItemType Directory -Force -Path $sslPath

# 生成自簽名證書
$cert = New-SelfSignedCertificate `
    -DnsName $domain `
    -CertStoreLocation cert:\LocalMachine\My `
    -NotAfter (Get-Date).AddDays(365) `
    -KeyAlgorithm RSA `
    -KeyLength 2048 `
    -HashAlgorithm SHA256 `
    -KeyUsage DigitalSignature, KeyEncipherment `
    -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.1")

# 導出證書和私鑰
$pwd = ConvertTo-SecureString -String "password" -Force -AsPlainText
Export-PfxCertificate -Cert "cert:\LocalMachine\My\$($cert.Thumbprint)" -FilePath $pfxFile -Password $pwd

# 使用 OpenSSL 轉換格式（需要安裝 OpenSSL）
$opensslCmd = @"
openssl pkcs12 -in $pfxFile -out $certFile -nodes -nokeys -password pass:password
openssl pkcs12 -in $pfxFile -out $keyFile -nodes -nocerts -password pass:password
"@

# 保存 OpenSSL 命令到批處理文件
$opensslCmd | Out-File -FilePath "convert-cert.bat" -Encoding ASCII

Write-Host "請在 Git Bash 中運行 convert-cert.bat 來完成證書轉換"
Write-Host "證書將保存在: $certFile"
Write-Host "私鑰將保存在: $keyFile" 