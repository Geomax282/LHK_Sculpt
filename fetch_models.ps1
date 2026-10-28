# Download the model weights (official sources, commercial use allowed)
# into .\models.
Set-Location $PSScriptRoot
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force models\birefnet, models\licenses | Out-Null

function Get-File($url, $out) {
    if (-not (Test-Path $out)) { Invoke-WebRequest $url -OutFile $out }
}

Get-File "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" models\depth_anything_v2_vits.pth
Get-File "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/model.safetensors" models\birefnet\model.safetensors
if (-not (Test-Path models\realesr-animevideov3-x2.bin)) {
    $tmp = Join-Path $env:TEMP "lhk_esrgan"
    New-Item -ItemType Directory -Force $tmp | Out-Null
    Invoke-WebRequest "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip" -OutFile "$tmp\esr.zip"
    Expand-Archive "$tmp\esr.zip" "$tmp\x" -Force
    Copy-Item (Get-ChildItem "$tmp\x" -Recurse -Filter realesr-animevideov3-x2.bin | Select-Object -First 1).FullName models\
    Remove-Item -Recurse -Force $tmp
}
Get-File "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/LICENSE" models\licenses\Depth-Anything-V2-Small-LICENSE.txt
Get-File "https://raw.githubusercontent.com/ZhengPeng7/BiRefNet/main/LICENSE" models\licenses\BiRefNet-LICENSE.txt
Get-File "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/LICENSE" models\licenses\Real-ESRGAN-LICENSE.txt
Write-Host "Modelle in $PSScriptRoot\models"
