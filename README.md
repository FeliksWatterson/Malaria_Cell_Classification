
$pattern = 'Windows Update Blocker|StopUpdates10|ShutUp10|Wu10Man|Win Updates Disabler|Winaero Tweaker|Wub_x64|Wub\.exe|OOSU10'

Write-Host "`n=== INSTALLED APPS ===" -ForegroundColor Cyan

$keys = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
    'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
)

Get-ItemProperty $keys -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -match $pattern } |
    Select-Object DisplayName, DisplayVersion, InstallLocation |
    Format-List

Write-Host "`n=== PORTABLE FILES ===" -ForegroundColor Cyan

$folders = @(
    "$env:USERPROFILE\Desktop",
    "$env:USERPROFILE\Downloads",
    "$env:USERPROFILE\Documents",
    "$env:PUBLIC\Desktop",
    "$env:ProgramFiles",
    "${env:ProgramFiles(x86)}",
    'C:\Tools'
) | Where-Object { $_ -and (Test-Path $_) }

Get-ChildItem -LiteralPath $folders -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match $pattern } |
    Select-Object FullName |
    Format-List

Write-Host "`n=== RELATED SERVICES ===" -ForegroundColor Cyan

Get-CimInstance Win32_Service |
    Where-Object {
        $_.Name -match 'StopUpdates|Wub|Wu10Man' -or
        $_.DisplayName -match 'StopUpdates|Windows Update Blocker|Wu10Man'
    } |
    Select-Object Name, DisplayName, State, StartMode |
    Format-List
