Get-ChildItem C:\ -Filter PsExec64.exe -Recurse -ErrorAction SilentlyContinue | Select-Object FullName
