# PowerShell script to copy fine-tuned model files

# Get the source directory from the first argument
param(
    [Parameter(Mandatory=$true)]
    [string]$SourceDir
)

# Check if source directory exists
if (-not (Test-Path -Path $SourceDir)) {
    Write-Host "Error: Source directory $SourceDir does not exist." -ForegroundColor Red
    exit 1
}

# Get the current script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item $ScriptDir).Parent.FullName

# Create the destination directory if it doesn't exist
$DestDir = Join-Path -Path $ProjectRoot -ChildPath "models\lora_weights"
if (-not (Test-Path -Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force
}

# Copy the files
try {
    $Files = Get-ChildItem -Path $SourceDir -File
    
    if ($Files.Count -eq 0) {
        Write-Host "Error: Source directory is empty." -ForegroundColor Red
        exit 1
    }
    
    foreach ($File in $Files) {
        $DestFile = Join-Path -Path $DestDir -ChildPath $File.Name
        Copy-Item -Path $File.FullName -Destination $DestFile -Force
        Write-Host "Copied: $($File.Name)" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Successfully copied model files to $DestDir" -ForegroundColor Green
    Write-Host "You can now run the application with your fine-tuned model." -ForegroundColor Cyan
}
catch {
    Write-Host "Error copying files: $_" -ForegroundColor Red
    exit 1
}