@echo off
REM MS SQL Server Connection Validation Script
REM Arguments: %1=host %2=port %3=database %4=user %5=password

SET HOST=%~1
SET PORT=%~2
SET DATABASE=%~3
SET USER=%~4
SET PASSWORD=%~5

REM Try to connect to MS SQL Server and create database if it doesn't exist
sqlcmd -S %HOST%,%PORT% -U %USER% -P %PASSWORD% -Q "IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = '%DATABASE%') CREATE DATABASE [%DATABASE%]" 2>nul

IF %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Connected to MS SQL Server and database is ready
    exit /b 0
) ELSE (
    echo ERROR: Failed to connect to MS SQL Server. Please check your credentials and ensure SQL Server is running.
    exit /b 1
)
