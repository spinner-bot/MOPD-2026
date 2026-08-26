# 下载 flash_attn wheel 文件
# 用法：在当前目录右键"使用 PowerShell 运行"，或在 PowerShell 里执行：.\download_flash_attn.ps1

$url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
$output = "flash_attn-2.8.3.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

Write-Host "开始下载 flash_attn wheel..."
Write-Host "URL: $url"
Write-Host "保存到: $output"

Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing

Write-Host "下载完成！文件大小："
Get-Item $output | Select-Object Name, Length
