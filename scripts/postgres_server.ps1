# Local PostgreSQL 17 for the BiBi review DB (portable install, no admin rights, no Windows service).
#   powershell -File scripts/postgres_server.ps1 start | stop | status | backup
# Server files: %USERPROFILE%\.bibi-postgres (binaries in pgsql\, data in data\, log in server.log)
# Passwords are not stored here; libpq reads them from %APPDATA%\postgresql\pgpass.conf.
param([ValidateSet('start','stop','status','backup')][string]$Command='status')
$ErrorActionPreference = 'Stop'
$Home_ = Join-Path $env:USERPROFILE '.bibi-postgres'
$Bin = Join-Path $Home_ 'pgsql\bin'
$Data = Join-Path $Home_ 'data'
$Log = Join-Path $Home_ 'server.log'
switch ($Command) {
  'start'  { & "$Bin\pg_ctl.exe" -D $Data -l $Log -o "-p 5432 -c listen_addresses=127.0.0.1" start }
  'stop'   { & "$Bin\pg_ctl.exe" -D $Data stop -m fast }
  'status' { & "$Bin\pg_ctl.exe" -D $Data status; & "$Bin\pg_isready.exe" -h 127.0.0.1 -p 5432 }
  'backup' {
    $root = Split-Path -Parent $PSScriptRoot
    $dir = Join-Path $root '2_DB\db\backups'
    New-Item -ItemType Directory -Force $dir | Out-Null
    $file = Join-Path $dir ("bibi_reviews_{0}.dump" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    & "$Bin\pg_dump.exe" -h 127.0.0.1 -p 5432 -U bibi -d bibi_reviews -Fc -f $file
    Get-Item $file | Select-Object FullName, Length
  }
}
